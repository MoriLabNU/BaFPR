import sys
import os 
import time
import copy
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import cv2

from models import _get_model
from solvers import structure_loss, _get_optim, adjust_lr, cross_entropy, mse_with_logits


import torch
import numpy as np 
import random
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_f
from utils import _get_datasetloader, AvgMeter, clip_gradient

from torch.utils.tensorboard import SummaryWriter






class Trainer():
    def __init__(self, config, logger_tuple: tuple):
        
        self.config = config
        self.scalar_writer, self.img_writer, self.logger = \
            logger_tuple
        self.current_epoch = 0
        self.training_epoch = self.config.runner.max_epoch

        # ! Handle model
        self.build_model_cfg()

        
        
        # ! get loss
        
        self.sup_criterion = mse_with_logits
        self.struc_criterion = structure_loss

        # ! load optimizer
        self.optimizer_cls = _get_optim(self.config)
        optim_args = copy.deepcopy(self.config.optimizer)
        optim_args.pop('type')
        
        self.optimizer = self.optimizer_cls(self.model.parameters(), **optim_args)
        if self.sub_model is not None:
            self.sub_optimizer = self.optimizer_cls(self.sub_model.parameters(), **optim_args)

        

        # TODO: add block for resuming training
        if self.config.runner['resume']:
            if self.main_process() and self.logger is not None:
                self.logger.info(f'loading model weights')
            checkpoint = torch.load(os.path.join(self.config.storage['save_dir'], 'log', 'ckp_latest.pt'), map_location=lambda storage, loc: storage.cuda())
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            if self.sub_model is not None:
                raise NotImplementedError('sub model resuming not implemented')
            self.current_epoch = checkpoint['epoch']
            if self.main_process() and self.logger is not None:
                self.logger.info(' resumed training state for latest checkpoint')

        
        # ! load data_loader and sampler 
        
        BuildDataloader = _get_datasetloader(self.config)
        self.train_labeled_loader, self.train_unlabeled_reader, self.test_loader_0,  self.test_loader_1, \
            self.test_loader_2, self.test_loader_3, self.test_loader_4 = BuildDataloader.build()

        
        # If using AMP
        
        self.scaler = GradScaler(enabled=True)
        if self.sub_model is not None:
           self.sub_scaler = GradScaler(enabled=True)


        

        # ! Get Metrics
        # * Train/Test: loss, mIoU, Acc
        # * @var avg_cost: 0~2: sup_loss, unsup_loss, val_loss
        # * @var avg_cost: 3~4:  mIoU for sup data, empty
        # * @var avg_cost: 5~6:  mIoU for unsup data, empty
        # * @var avg_cost: 7~9:  mIoU for val data, empty, empty
        self.avg_cost = np.zeros((self.training_epoch, 10))
        
        self.iter_per_epoch = None # * setting according to sampler
        
        # * @var time_count: data loading time per batch, computation cost per batch
        self.time_count = np.zeros((self.training_epoch, 2))

    def build_model_cfg(self):
        # ! Handle model
        self.model_cls = _get_model(self.config)
        
        self.model = self.model_cls(**self.config.model['args'])
        
        if self.config.get('enable_polar_consist', False):
            self.sub_model = self.model_cls(**self.config.model['args'])
        else:
            self.sub_model = None
        if self.main_process() and self.logger is not None:
            self.logger.info(f'Built new model')
        if dist.get_world_size() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(),  find_unused_parameters=True)

        else:
            self.model.cuda()
        if self.sub_model is not None:
            if dist.get_world_size() > 1:
                self.sub_model =  torch.nn.parallel.DistributedDataParallel(self.sub_model.cuda(),  find_unused_parameters=True)
            else:
                self.sub_model.cuda()



    
    
    @staticmethod
    def get_mIoU(confusion_matrix):
        assert isinstance(confusion_matrix, torch.Tensor)
        num_classes = confusion_matrix.shape[0]
        intersection = torch.diag(confusion_matrix)
        union = torch.sum(confusion_matrix, dim=0) + torch.sum(confusion_matrix, dim=1) + 0.00001
        mIoU = intersection / (union - intersection )
        return mIoU


    @staticmethod
    def get_Dice2(pred, target):
        pred = pred.flatten()
        target = target.flatten()
        intersection = pred * target
        union = torch.sum(pred) + torch.sum(target) + 0.1
        dice_co = 2*intersection.sum() / union.sum()
        return dice_co



    @staticmethod
    def denormalize(img):
        x = transforms_f.normalize(img, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x 

    @staticmethod
    def update_confusion_mat(confusion_matrix, pred, target):
        assert isinstance(confusion_matrix, torch.Tensor)
        num_classes = confusion_matrix.shape[0]
        with torch.no_grad():
            k = (target >= 0) & (target < num_classes)
            inds = num_classes * target[k].to(torch.int64) + pred[k].to(torch.int64)
            confusion_matrix += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return confusion_matrix

        


    
    
    def train_step(self):
        max_it = -1 if self.iter_per_epoch is None else self.iter_per_epoch

        
        
        self.confusion_mat = torch.zeros((self.config.model.args['num_classes'], \
            self.config.model.args['num_classes'])).cuda()

        self.model.train()
        if self.sub_model is not None:
            self.sub_model.train()
        
        end = time.time()
        total_iter = len(self.train_labeled_loader)

        size_rates = [0.75, 1, 1.25]
        loss_P2_record = AvgMeter()
        loss_be_record = AvgMeter()
        loss_consist_record = AvgMeter()

        

        
        for i_iter, (image_dict, file_path) in enumerate(self.train_labeled_loader):

            
            for rate in size_rates:
            
            
            
                img = image_dict['aug_image']
                label = image_dict['aug_label']

                
                
                if (i_iter >= max_it) and (max_it > 0):
                    break
                
                self.time_count[self.current_epoch, 0] =+ (time.time() -end) / total_iter
                end = time.time()
                

                img, label = img.cuda().float(), label.cuda().float()
                if rate != 1:
                    trainsize = int(round(352 * rate / 32) * 32)
                    img = F.interpolate(img, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    label  = F.interpolate(label.unsqueeze(1),  size=(trainsize, trainsize), mode='nearest').squeeze(1)


                # ! supervised loss
                
                
                with autocast(enabled=True):
                    
                    pred_l_1, pred_l_2, dist_est = self.model(img)
                    
                
                pred_l_large_1 = torch.nn.functional.interpolate(pred_l_1, size = label.shape[1:],\
                    mode = 'bilinear', align_corners = True)
                if pred_l_2 is not None:
                    pred_l_large_2 = torch.nn.functional.interpolate(pred_l_2, size = label.shape[1:],\
                        mode = 'bilinear', align_corners = True)
                else:
                    pred_l_large_2 = None
                if dist_est is not None:
                    pred_dist_large = torch.nn.functional.interpolate(dist_est, size = label.shape[1:],\
                        mode = 'bilinear', align_corners = True)


                
                if self.sub_model is not None:
                    
                
                    if self.config.get('enable_polar_consist', False):
                        img_sub = image_dict['aug_image_polar']
                        label_sub = image_dict['aug_label_polor']
                        center_list = image_dict['polar_center']
                    else:
                        img_sub = image_dict['aug_image']
                        label_sub = image_dict['aug_label']
                    
                    if rate != 1:
                        img_sub = F.interpolate(img_sub, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        label_sub  = F.interpolate(label_sub.unsqueeze(1),  size=(trainsize, trainsize), mode='nearest').squeeze(1)
                    
                    
                    
                    with autocast(enabled=True):
                        img_sub, label_sub = img_sub.cuda(), label_sub.cuda()
                        pred_l_sub_1, pred_l_sub_2, _ = self.sub_model(img_sub)
                        pred_l_large_sub_1 = torch.nn.functional.interpolate(pred_l_sub_1, size = label_sub.shape[1:],\
                            mode = 'bilinear', align_corners = True)
                        pred_l_large_sub_2 = torch.nn.functional.interpolate(pred_l_sub_2, size = label_sub.shape[1:],\
                            mode = 'bilinear', align_corners = True)

                
                class_weight = None
                with autocast(enabled=True):
                    sup_loss_P1 = structure_loss(pred_l_large_1, label.unsqueeze(1))
                    if pred_l_large_2 is not None:
                        sup_loss_P2 = structure_loss(pred_l_large_2, label.unsqueeze(1))
                    else:
                        sup_loss_P2 = 0.0
                    sup_loss = sup_loss_P1 + sup_loss_P2
                    if self.sub_model is not None:
                        sup_loss_sub_P1 = structure_loss(pred_l_large_sub_1, label_sub.unsqueeze(1))
                        sup_loss_sub_P2 = structure_loss(pred_l_large_sub_2, label_sub.unsqueeze(1))
                        sup_loss_sub = sup_loss_sub_P1 + sup_loss_sub_P2



                    if self.config.get('boundary_prior', False):
                        with torch.no_grad():
                            label_4_dis = label.clone().cpu().numpy()
                            distance_weighting = []
                            for _ in range(label.shape[0]):
                                
                                boundary = (skimage_seg.find_boundaries(label_4_dis[_], mode='inner') == 0).astype(np.uint8)
                                dis_map = distance(boundary) 
                                
                                dis_map = (1 - dis_map / np.max(dis_map)) 
                                distance_weighting.append(dis_map)
                            distance_weighting = torch.from_numpy(np.stack(distance_weighting)).to(pred_l_large_1.device)
                        
                        
                        



                    
                    if self.config.get('enable_polar_consist', False) and self.current_epoch >=0:
                        with torch.no_grad():

                            pred_l_large_sub_np = pred_l_large_sub_1.sigmoid().squeeze().cpu().numpy()
                            pred_polar2cart = []
                            for _ in range(pred_l_large_sub_np.shape[0]):
                                center = (int(center_list[0][_].item()), int(center_list[1][_].item()))
                                i_pred = self.test_loader_0.dataset.inverse_polar_trans((pred_l_large_sub_np[_] > 0.5).astype(np.uint8), \
                                        center = center)
                                pred_polar2cart.append(i_pred)
                            pred_polar2cart = torch.from_numpy(np.stack(pred_polar2cart)).cuda()


                            pred_l_cart_large_np = pred_l_large_1.sigmoid().squeeze().cpu().numpy()
                            pred_cart2polar = []
                            for _ in range(pred_l_cart_large_np.shape[0]):
                                center = (int(center_list[0][_].item()), int(center_list[1][_].item()))
                                i_pred = self.test_loader_0.dataset.polor_trans((pred_l_cart_large_np[_] > 0.5).astype(np.uint8), \
                                            center = center)
                                pred_cart2polar.append(i_pred)
                            pred_cart2polar = torch.from_numpy(np.stack(pred_cart2polar)).cuda()
                        
                        consist_loss =  self.sup_criterion(pred_l_large_2, pred_polar2cart.unsqueeze(1)) 
                        consist_loss_sub = self.sup_criterion(pred_l_large_sub_2, pred_cart2polar.unsqueeze(1)) 

                        
                        if self.config.get('boundary_prior', False):
                            
                            be_loss = self.sup_criterion(pred_dist_large, distance_weighting.unsqueeze(1).float())


                    else:
                        be_loss = torch.zeros(1).cuda()
                        consist_loss = torch.zeros(1).cuda()
                        sup_loss_sub = torch.zeros(1).cuda()
                        be_loss_sub = torch.zeros(1).cuda()
                        consist_loss_sub = torch.zeros(1).cuda()
                        
                        if self.config.get('boundary_prior', False):
                            
                            with autocast(enabled=True):
                                be_loss = self.sup_criterion(pred_dist_large, distance_weighting.unsqueeze(1).float())
                        
                with autocast(enabled=True):
                    
                    consist_weight = self.config.get('consist_weight', 2)
                    if  self.config.get('disable_be', False):
                        total_loss = sup_loss   #+ unsup_loss + point_loss 
                    elif  self.config.get('disable_sup', False):
                        total_loss =  be_loss + consist_loss * consist_weight
                    else:
                        
                        total_loss = sup_loss  +  be_loss + consist_loss * consist_weight 

                
                if rate == 1:
                    loss_P2_record.update(sup_loss.data, self.config.data.samples_per_gpu)
                    loss_be_record.update(be_loss.data, self.config.data.samples_per_gpu)
                    loss_consist_record.update(consist_loss.data, self.config.data.samples_per_gpu)
                    # print(f'sup_loss: {sup_loss.item()}, uncert_loss: {be_loss.item()}, consist_loss: {consist_loss.item()}')

                self.optimizer.zero_grad()
                if self.sub_model is not None:
                    self.sub_optimizer.zero_grad()

                
                self.scaler.scale(total_loss).backward()
                clip_gradient(self.optimizer, self.config.optimizer_config.grad_clip)
                self.scaler.step(self.optimizer)
                

                if self.sub_model is not None:
                    
                    with autocast(enabled=True):
 
                        total_loss_sub = sup_loss_sub 
                    self.sub_scaler.scale(total_loss_sub).backward()
                    clip_gradient(self.sub_optimizer, self.config.optimizer_config.grad_clip)
                    self.sub_scaler.step(self.sub_optimizer)
                    self.sub_scaler.update()
                self.scaler.update()


                self.confusion_mat = self.update_confusion_mat(self.confusion_mat, pred_l_large_1.argmax(1).flatten(),\
                    label.flatten())
                

                self.avg_cost[self.current_epoch, 0] += total_loss.detach().item() / total_iter
                
                self.time_count[self.current_epoch, 1] += (time.time() - end)/ total_iter

        if self.logger is not None:
            self.logger.info(f'Epoch, {self.current_epoch}, seg avg_loss: {loss_P2_record.show()}')
            self.logger.info(f'Epoch, {self.current_epoch}, boundary prior avg_loss: {loss_be_record.show()}')
            self.logger.info(f'Epoch, {self.current_epoch}, consistency avg_loss: {loss_consist_record.show()}')
        else:
            print(f'Epoch, {self.current_epoch}, seg avg_loss: {loss_P2_record.show()}')
            print(f'Epoch, {self.current_epoch}, boundary prior avg_loss: {loss_be_record.show()}')
            print(f'Epoch, {self.current_epoch}, consistency avg_loss: {loss_consist_record.show()}')


            

    def train(self, percent=0):

        
        self.local_writer = SummaryWriter(log_dir=os.path.join(self.config.save_to, 'log'), \
            filename_suffix=f'._monitoring') if self.config.save_to is not None else None
        for i_epoch in range(self.current_epoch, self.training_epoch):
            print(f'training epoch {i_epoch}')
            self.current_epoch = i_epoch
            percent += 1

            if self.ifdist():
                self.train_labeled_loader.sampler.set_epoch(i_epoch)



            adjust_lr(self.optimizer, self.config.optimizer.lr, i_epoch, 0.1, 200)
            if self.sub_model is not None:
                adjust_lr(self.sub_optimizer, self.config.optimizer.lr, i_epoch, 0.1, 200)
            current_lr = self.optimizer.param_groups[0]['lr']


            
            self.train_step()
            

            mIoU = self.get_mIoU(self.confusion_mat)
            self.avg_cost[self.current_epoch, 3] = torch.mean(mIoU).detach().item()
            

            if self.main_process() and ((i_epoch+1) % self.config.evaluation['interval'] == 0 or (i_epoch+1)==self.training_epoch):
                val_confusion_mat, val_cost, val_confusion_mat_sub, val_confusion_mat_ensem = self.eval(percent)
                self.val_mIoU = self.get_mIoU(val_confusion_mat)
               
                if self.logger is not None:
                    self.logger.info(f'val IoU per class: {self.val_mIoU.detach().cpu().numpy()} at epoch {self.current_epoch}')
                    self.logger.info(f'val cost: {val_cost} at epoch {self.current_epoch}')

                if self.sub_model is not None:
                    val_mIoU_sub = self.get_mIoU(val_confusion_mat_sub)
                    val_mIoU_ensem = self.get_mIoU(val_confusion_mat_ensem)
                    if self.logger is not None:
                        self.logger.info(f'val_sub IoU per class: {val_mIoU_sub.detach().cpu().numpy()} at epoch {self.current_epoch}')
                        self.logger.info(f'val_ensem IoU per class: {val_mIoU_ensem.detach().cpu().numpy()} at epoch {self.current_epoch}')

                
                if self.local_writer is not None:
                    self.local_writer.add_scalar(f'val/sup_mIoU', torch.mean(self.val_mIoU).detach().item(), i_epoch)
                    self.local_writer.add_scalar(f'val/sup_loss', val_cost[0], i_epoch)
                    if self.sub_model is not None:
                        self.local_writer.add_scalar(f'val/sup_mIoU_sub', torch.mean(val_mIoU_sub).detach().item(), i_epoch)
                        self.local_writer.add_scalar(f'val/sup_mIoU_ensem', torch.mean(val_mIoU_ensem).detach().item(), i_epoch)


                    

                if self.local_writer is not None:
                    self.local_writer.add_scalar(f'train/sup_mIoU', self.avg_cost[self.current_epoch, 3], i_epoch)
                    self.local_writer.add_scalar(f'train/sup_loss', self.avg_cost[self.current_epoch, 0], i_epoch)
                    self.local_writer.add_scalar(f'train/unsup_loss', self.avg_cost[self.current_epoch, 1], i_epoch)
                    self.local_writer.add_scalar(f'train/lr', current_lr, i_epoch)

    
    # TODO: eval function
    @torch.no_grad()
    def eval(self, percent=0):
        if self.main_process() and self.logger is not None:
            self.logger.info(f'valdation at epoch {self.current_epoch}')
        self.model.eval()
        if self.sub_model is not None:
            self.sub_model.eval()

        val_confusion_mat = torch.zeros((self.config.model.args['num_classes'], \
            self.config.model.args['num_classes'])).cuda()

        val_confusion_mat_sub = torch.zeros((self.config.model.args['num_classes'], \
            self.config.model.args['num_classes'])).cuda()
        
        val_confusion_mat_ensem = torch.zeros((self.config.model.args['num_classes'], \
            self.config.model.args['num_classes'])).cuda()

        val_cost = np.zeros((3))
        dataset_names = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
        for i_dataset, test_loader in enumerate((self.test_loader_0, self.test_loader_1, self.test_loader_2, self.test_loader_3, self.test_loader_4)):
            total_val_iter = len(test_loader)
            vis_count = 0
            DSC = 0.0
            DSC_sub = 0.0
            DSC_ens = 0.0
            if self.config.save_to is not None:
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', f'{dataset_names[i_dataset]}'), exist_ok=True)
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', 'single', f'{dataset_names[i_dataset]}'), exist_ok=True)
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', 'ensem', f'{dataset_names[i_dataset]}' ), exist_ok=True)
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', 'dist', f'{dataset_names[i_dataset]}'), exist_ok=True)

            for i_iter, (image_dict, file_path) in enumerate(test_loader):
                    
                img_cart = image_dict['aug_image']
                label_cart = image_dict['aug_label']
                

                if self.model is not None:

                    img_cart, label_cart = img_cart.cuda(), label_cart.cuda()
                    pred_l_cart_1, pred_l_cart_2, dist_val = self.model(img_cart)

                    pred_l_cart_large = torch.nn.functional.interpolate(pred_l_cart_1, size = label_cart.shape[1:],\
                    mode = 'bilinear', align_corners = True)
                    if pred_l_cart_2 is not None:
                        pred_l_cart_large_2 = torch.nn.functional.interpolate(pred_l_cart_2, size = label_cart.shape[1:],\
                        mode = 'bilinear', align_corners = True)
                    else:
                        pred_l_cart_large_2 = pred_l_cart_large
                        pred_l_cart_2 = pred_l_cart_1

                    pred_l_large_np_ensem = ((pred_l_cart_large+pred_l_cart_large_2)/2).cpu().numpy()

                    pred_gt_cart = ((pred_l_cart_large + pred_l_cart_large_2) / 2).sigmoid() > 0.5
                    center_list_pred = (torch.zeros(pred_l_cart_large.shape[0]), torch.zeros(pred_l_cart_large.shape[0]))
                    
                    

                    
                    dice = self.get_Dice2(pred_gt_cart, label_cart)
                    DSC += dice

                    if (self.current_epoch+1)==self.training_epoch and self.config.save_to is not None:
                        
                        res = torch.nn.functional.interpolate(pred_l_cart_1 + pred_l_cart_2, size = label_cart.shape[1:],\
                            mode = 'bilinear', align_corners = True)
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        img_name = file_path[0].split('/')[-1]
                        if img_name.endswith('.jpg'):
                            img_name = img_name.split('.jpg')[0] + '.png'
                        cv2.imwrite(os.path.join(self.config.save_to, 'log', 'results', 'single', f'{dataset_names[i_dataset]}',  f'{img_name}'), res*255)
                        if dist_val is not None:
                            dist_val = torch.nn.functional.interpolate(dist_val, size = label_cart.shape[1:],\
                                mode = 'bilinear', align_corners = True)
                            dist_val = dist_val.sigmoid().data.cpu().numpy().squeeze()
                            cv2.imwrite(os.path.join(self.config.save_to, 'log', 'results',  'dist', f'{dataset_names[i_dataset]}', f'{img_name}'), dist_val*255)


                    

                    if self.sub_model is not None:

                        for i_center in range(pred_gt_cart.shape[0]):
                            cX, cY = test_loader.dataset.centroid(pred_gt_cart[i_center].squeeze().cpu().numpy().astype(np.float32))
                            center_list_pred[0][i_center] =+ cX
                            center_list_pred[1][i_center] =+ cY


                        img = np.zeros((img_cart.shape))
                        for i_polar in range(img.shape[0]):
                            img[i_polar] += test_loader.dataset.polor_trans(img_cart[i_polar].permute(1,2,0).cpu().numpy(), center = (int(center_list_pred[0][i_polar].item()), int(center_list_pred[1][i_polar].item()))).transpose((2,0,1))
                        img = torch.from_numpy(img).cuda().float()

                        with autocast(enabled=True):
                            img = img.cuda()
                            pred_l_1, pred_l_2, _ = self.sub_model(img)

                            
                            
                        pred_l_large = torch.nn.functional.interpolate((pred_l_1+pred_l_2)/2, size = label_cart.shape[1:],\
                            mode = 'bilinear', align_corners = True)
                        
                        

                # ! Main Process Only
                sup_loss = 0


                #for polar images
                if self.sub_model is not None:

                    pred_l_large_np = np.zeros((pred_l_large.shape[0], pred_l_large.shape[2], pred_l_large.shape[3]))
                    

                    for i, i_img in enumerate(img):
                        
                        center = (int(center_list_pred[0][i].item()), int(center_list_pred[1][i].item()))
                        i_pred = test_loader.dataset.inverse_polar_trans(pred_l_large[i].squeeze().cpu().numpy(), \
                            center = center)

                        pred_l_large_np_ensem[i] += test_loader.dataset.inverse_polar_trans(pred_l_large[i].squeeze().cpu().numpy(), \
                            center = center)
                        
                        
                        pred_l_large_np[i] += i_pred
                        
                    pred_l_large_np = torch.from_numpy(pred_l_large_np).cuda().sigmoid() > 0.5
                    pred_l_large_np_ensem_pred_gt = torch.from_numpy(pred_l_large_np_ensem).cuda().sigmoid() > 0.5

                    dice = self.get_Dice2(pred_l_large_np_ensem_pred_gt, label_cart)
                    DSC_ens+=dice


                    if (self.current_epoch+1)==self.training_epoch and self.config.save_to is not None:
                        
                        res = torch.from_numpy(pred_l_large_np_ensem).sigmoid().data.cpu().numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        img_name = file_path[0].split('/')[-1]
                        if img_name.endswith('.jpg'):
                            img_name = img_name.split('.jpg')[0] + '.png'
                        cv2.imwrite(os.path.join(self.config.save_to, 'log', 'results', 'ensem', f'{dataset_names[i_dataset]}',  f'{img_name}'), res*255)


                    


                # ! tensorboard
                
                if vis_count < self.config.log_config['img2save']:
                    if self.config.data.get('polar', False):
                        cX, cY = center_list_pred[0][0].item(), center_list_pred[1][0].item()
                        label2write = (label_cart[0]*255).cpu().numpy()
                        pred2write = ((pred_l_cart_large[0].argmax(0)).unsqueeze(-1).cpu().numpy().astype(np.uint8) * 255)
                        img2write =  self.denormalize(img_cart[0]).permute((1,2,0)).cpu().numpy()*255

                        self.img_writer.add_image(
                        "{}_1_label".format(vis_count), label2write[...,np.newaxis],
                                    global_step=percent, dataformats="HWC")
                        self.img_writer.add_image(
                            "{}_1_pred".format(vis_count), pred2write,
                                    global_step=percent, dataformats="HWC")
                        self.img_writer.add_image(
                            "{}_1_img".format(vis_count),img2write,
                                    global_step=percent, dataformats="HWC")


                        if self.sub_model is not None:
                            self.img_writer.add_image(
                                "{}_1_pred_sub".format(vis_count), ((pred_l_large_np[0]).unsqueeze(-1).cpu().numpy().astype(np.uint8) * 255),
                                        global_step=percent, dataformats="HWC")


                            
                            
                    else:
                        self.img_writer.add_image(
                        "{}_1_label".format(vis_count), (label_cart[0]*255).unsqueeze(-1).cpu().numpy(),
                                    global_step=percent, dataformats="HWC")
                        self.img_writer.add_image(
                            "{}_1_pred".format(vis_count), ((pred_l_cart_large[0].argmax(0)).unsqueeze(-1).cpu().numpy().astype(np.uint8) * 255),
                                    global_step=percent, dataformats="HWC")
                        self.img_writer.add_image(
                            "{}_1_img".format(vis_count), self.denormalize(img_cart[0]).permute((1,2,0)).cpu().numpy(),
                                    global_step=percent, dataformats="HWC")
                    vis_count+=1
            if self.logger is not None:
                self.logger.info(f'epoch: {self.current_epoch}, the {i_dataset} dataset, val_dice: {DSC / total_val_iter}, ensem: {DSC_ens / total_val_iter}')
            else:
                print(f'epoch: {self.current_epoch}, the {i_dataset} dataset, val_dice: {DSC / total_val_iter}, ensem: {DSC_ens / total_val_iter}')
        if self.config.save_to is not None:
            filename = self.config.save_to + '/train_epoch_' + str(self.current_epoch) + '.pth'
            self.logger.info('Saving checkpoint to: ' + filename)
            torch.save({'percent': percent, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                       filename)
            if self.sub_model is not None:
                torch.save({'percent': percent, 'state_dict': self.sub_model.state_dict(), 'optimizer': self.sub_optimizer.state_dict()},
                       filename + 'sub.pth')
                

        
        return val_confusion_mat, val_cost, val_confusion_mat_sub, val_confusion_mat_ensem

    def main_process(self):
        rank = int(os.environ['RANK'])
        return True if rank<=0 else False

    def ifdist(self):
        return True if dist.get_world_size() > 1 else False

   


