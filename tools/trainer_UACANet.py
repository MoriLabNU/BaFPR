import sys
# sys.path.append('..')
import os 
import time
import copy
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import cv2

from models import _get_model
from solvers import structure_loss, _get_optim, adjust_lr, cross_entropy
#from optim import _get_loss, _get_optimizer, _get_scheduler
#from datasets import _get_datasetloader
#from data.data_mixture import classmix
#from optim.losses import smooth_CE

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
        
        self.sup_criterion = cross_entropy
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
        self.model.cuda()
        if self.sub_model is not None:
            self.sub_model.cuda()
        # if self.config.distributed['multiprocessing_distributed']:
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(), 
        #     device_ids=[self.config.trainer['gpu']], find_unused_parameters=True)

    
    
    @staticmethod
    def get_mIoU(confusion_matrix):
        assert isinstance(confusion_matrix, torch.Tensor)
        num_classes = confusion_matrix.shape[0]
        intersection = torch.diag(confusion_matrix)
        union = torch.sum(confusion_matrix, dim=0) + torch.sum(confusion_matrix, dim=1) + 0.00001
        mIoU = intersection / (union - intersection )
        return mIoU

    @staticmethod
    def get_dice_co(confusion_matrix):
        assert isinstance(confusion_matrix, torch.Tensor)
        num_classes = confusion_matrix.shape[0]
        intersection = torch.diag(confusion_matrix)
        union = torch.sum(confusion_matrix, dim=0) + torch.sum(confusion_matrix, dim=1) + 0.1
        dice_co = 2*intersection / (union)
        return dice_co

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

            #import pdb; pdb.set_trace()
            for rate in size_rates:
            
            
            
                img = image_dict['aug_image']
                label = image_dict['aug_label']

                #
                
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
                    #import pdb; pdb.set_trace()
                    pred, sup_loss, debug = self.model(img, label)
                    


                
                class_weight = None
                with autocast(enabled=True):
                    sup_loss = sup_loss
                        
                        



                

                   
                        
                with autocast(enabled=True):
                    total_loss = sup_loss

                #import pdb; pdb.set_trace()
                if rate == 1:
                    loss_P2_record.update(sup_loss.data, self.config.data.samples_per_gpu)
                    

                self.optimizer.zero_grad()

                
                self.scaler.scale(total_loss).backward()
                clip_gradient(self.optimizer, self.config.optimizer_config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()


                # if self.config.distributed['multiprocessing_distributed']:
                #     dist.all_reduce(total_loss.detach())
                #     dist.all_reduce(unsup_loss.detach())
                #     dist.all_reduce(self.confusion_mat)

                self.avg_cost[self.current_epoch, 0] += total_loss.detach().item() / total_iter
                #self.avg_cost[self.current_epoch, 1] += unsup_loss.detach().item() / total_iter
                self.time_count[self.current_epoch, 1] += (time.time() - end)/ total_iter

        if self.logger is not None:
            self.logger.info(f'Epoch, {self.current_epoch}, avg_loss: {loss_P2_record.show()}')
            #self.logger.info(f'Epoch, {self.current_epoch}, avg_loss: {loss_be_record.show()}')
            #self.logger.info(f'Epoch, {self.current_epoch}, avg_loss: {loss_consist_record.show()}')
        else:
            print(f'Epoch, {self.current_epoch}, avg_loss: {loss_P2_record.show()}')
            #print(f'Epoch, {self.current_epoch}, avg_loss: {loss_be_record.show()}')
            #print(f'Epoch, {self.current_epoch}, avg_loss: {loss_consist_record.show()}')


            

    def train(self, percent=0):

        
        self.local_writer = SummaryWriter(log_dir=os.path.join(self.config.save_to, 'log'), \
            filename_suffix=f'._monitoring') if self.config.save_to is not None else None
        for i_epoch in range(self.current_epoch, self.training_epoch):
            print(f'training epoch {i_epoch}')
            self.current_epoch = i_epoch
            percent += 1

            if self.ifdist():
                self.train_labeled_loader.sampler.set_epoch(i_epoch)
                # if self.config.trainer.train_unlabeled:
                #     self.train_unlabeled_reader.dataloader.sampler.set_epoch(i_epoch)

            # if i_epoch+1 in [64, 96] and self.config.model.name == 'SAN':
            #     self.optimizer.param_groups[0]['lr'] *= 0.5
            #     self.optimizer.param_groups[1]['lr'] *= 0.5
            #     if self.sub_optimizer is not None:
            #         self.sub_optimizer.param_groups[0]['lr'] *= 0.5
            #         self.sub_optimizer.param_groups[1]['lr'] *= 0.5
            adjust_lr(self.optimizer, self.config.optimizer.lr, i_epoch, 0.1, 200)

            current_lr = self.optimizer.param_groups[0]['lr']

            # if self.config.distributed['multiprocessing_distributed']:
            #     self.train_labeled_loader.sampler.set_epoch(i_epoch)
            #     self.train_unlabeled_reader.dataloader.sampler.set_epoch(i_epoch)
            
            self.train_step()
            
            # self.scheduler.step()
            # if self.sub_model is not None:
            #     self.sub_scheduler.step()
            mIoU = self.get_mIoU(self.confusion_mat)
            self.avg_cost[self.current_epoch, 3] = torch.mean(mIoU).detach().item()
            

            if self.main_process() and ((i_epoch+1) % self.config.evaluation['interval'] == 0 or (i_epoch+1)==self.training_epoch):
                val_confusion_mat, val_cost, val_confusion_mat_sub, val_confusion_mat_ensem = self.eval(percent)
                # self.val_mIoU = self.get_mIoU(val_confusion_mat)
               

                


            # ! record lr if necessary
            # if self.main_process():
            #     if self.logger is not None:
            #         self.logger.info(f'current lr: {current_lr}')
            #         self.logger.info(f'finishing epoch: {i_epoch}')
                    
        # if self.main_process():
        #     try:
        #         self.scalar_writer.add_scalar('Active/global_mIoU', self.val_mIoU.mean().detach().item(), global_step=percent)
        #     except AttributeError:
        #         pass
    
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
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', f'{dataset_names[i_dataset]}', 'single'), exist_ok=True)
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', f'{dataset_names[i_dataset]}', 'ensem'), exist_ok=True)
                os.makedirs(os.path.join(self.config.save_to, 'log', 'results', f'{dataset_names[i_dataset]}', 'dist'), exist_ok=True)

            for i_iter, (image_dict, file_path) in enumerate(test_loader):
                    
                img_cart = image_dict['aug_image']
                label_cart = image_dict['aug_label']
                

                if self.model is not None:

                    img_cart, label_cart = img_cart.cuda(), label_cart.cuda()
                    pred_l_cart_1, _, _ = self.model(img_cart)

                    pred_l_cart_large = torch.nn.functional.interpolate(pred_l_cart_1, size = label_cart.shape[1:],\
                    mode = 'bilinear', align_corners = True)



                    pred_gt_cart = ((pred_l_cart_large ) / 2).sigmoid() > 0.5
                    center_list_pred = (torch.zeros(pred_l_cart_large.shape[0]), torch.zeros(pred_l_cart_large.shape[0]))
                    
                    dice = self.get_Dice2(pred_gt_cart, label_cart)
                    DSC += dice

                    if (self.current_epoch+1)==self.training_epoch and self.config.save_to is not None:
                        
                        res = torch.nn.functional.interpolate(pred_l_cart_1, size = label_cart.shape[1:],\
                            mode = 'bilinear', align_corners = True)
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        img_name = file_path[0].split('/')[-1]
                        if img_name.endswith('.jpg'):
                            img_name = img_name.split('.jpg')[0] + '.png'
                        cv2.imwrite(os.path.join(self.config.save_to, 'log', 'results', f'{dataset_names[i_dataset]}', 'single', f'{img_name}'), res*255)
                        


                # ! Main Process Only
                sup_loss = 0



                # ! tensorboard
                
            if self.logger is not None:
                self.logger.info(f'epoch: {self.current_epoch}, the {i_dataset} dataset, val_dice: {DSC / total_val_iter}, ensem: {DSC_ens / total_val_iter}')
            else:
                print(f'epoch: {self.current_epoch}, the {i_dataset} dataset, val_dice: {DSC / total_val_iter}, ensem: {DSC_ens / total_val_iter}')
        if self.config.save_to is not None:
            filename = self.config.save_to + '/train_epoch_' + str(self.current_epoch) + '.pth'
            self.logger.info('Saving checkpoint to: ' + filename)
            torch.save({'percent': percent, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                       filename)
                

        
        return val_confusion_mat, val_cost, val_confusion_mat_sub, val_confusion_mat_ensem

    def main_process(self):
        rank = int(os.environ['RANK'])
        return True if rank<=0 else False

    def ifdist(self):
        return True if dist.get_world_size() > 1 else False

   


