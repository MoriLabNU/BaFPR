import random
import torch
import os
from .datasets import BuildDataset
import numpy as np
import torch.distributed as dist

class DataReader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)
    
    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            self.construct_iter()
            return self.dataloader_iter.next()

class BuildDataloader:
    def __init__(self, config):
        self.config = config
        
        


    def build(self):
        
        top_root = self.config.data_path
        
        train_dir = os.path.join(top_root, 'TrainDataset', 'images')
        val_dir_0 =  os.path.join(top_root, 'TestDataset', 'CVC-300', 'images')
        val_dir_1 =  os.path.join(top_root, 'TestDataset', 'CVC-ClinicDB', 'images')
        val_dir_2 =  os.path.join(top_root, 'TestDataset', 'CVC-ColonDB', 'images')
        val_dir_3 =  os.path.join(top_root, 'TestDataset', 'ETIS-LaribPolypDB', 'images')
        val_dir_4 =  os.path.join(top_root, 'TestDataset', 'Kvasir', 'images')

        crop_size = (352,352)

        
        self.train_idx = [ train_dir + '/' + f for f in os.listdir(train_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.val_idx_0 = [val_dir_0 + '/' + f for f in os.listdir(val_dir_0) if f.endswith('.tif') or f.endswith('.png')]
        self.val_idx_1 = [val_dir_1 + '/' + f for f in os.listdir(val_dir_1) if f.endswith('.tif') or f.endswith('.png')]
        self.val_idx_2 = [val_dir_2 + '/' + f for f in os.listdir(val_dir_2) if f.endswith('.tif') or f.endswith('.png')]
        self.val_idx_3 = [val_dir_3 + '/' + f for f in os.listdir(val_dir_3) if f.endswith('.tif') or f.endswith('.png')]
        self.val_idx_4 = [val_dir_4 + '/' + f for f in os.listdir(val_dir_4) if f.endswith('.tif') or f.endswith('.png')]
        
            



        self.train_labeled_dataset = BuildDataset( idx=self.train_idx, \
            config=self.config, \
            crop_size=crop_size[0], \
            scale_size=None, 
            training=True, augmentatation=False)

        # self.train_unlabel_dataset = BuildDataset(self.dataset, idx=self.train_u_idx, \
        #     config=self.config, \
        #     crop_size=self.config.dataset['args']['crop_size'], \
        #     scale_size=self.config.dataset['args']['scale_size'], 
        #     training=True, augmentatation=False)
        
        self.test_dataset_0 = BuildDataset( idx=self.val_idx_0, \
            config=self.config, \
            crop_size=crop_size[0], \
            scale_size=None,
            training=False, augmentatation=False)

        self.test_dataset_1 = BuildDataset( idx=self.val_idx_1, \
            config=self.config, \
            crop_size=crop_size[0], \
            scale_size=None,
            training=False, augmentatation=False)
        
        self.test_dataset_2 = BuildDataset( idx=self.val_idx_2, \
            config=self.config, \
            crop_size=crop_size[0], \
            scale_size=None,
            training=False, augmentatation=False)
        
        self.test_dataset_3 = BuildDataset( idx=self.val_idx_3, \
            config=self.config, \
            crop_size=crop_size[0], \
            scale_size=None,
            training=False, augmentatation=False)
        
        self.test_dataset_4 = BuildDataset( idx=self.val_idx_4, \
            config=self.config, \
            crop_size=crop_size[0], \
            scale_size=None,
            training=False, augmentatation=False)

        try:
            dist.get_world_size()

            if dist.get_world_size() > 1:
                print(f'world size {dist.get_world_size()}')
                train_labeled_sampler = torch.utils.data.distributed.DistributedSampler(self.train_labeled_dataset)
                train_unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(self.train_unlabel_dataset)
                test_sampler_0 = torch.utils.data.distributed.DistributedSampler(self.test_dataset_0)
                test_sampler_1 = torch.utils.data.distributed.DistributedSampler(self.test_dataset_1)
                test_sampler_2 = torch.utils.data.distributed.DistributedSampler(self.test_dataset_2)
                test_sampler_3 = torch.utils.data.distributed.DistributedSampler(self.test_dataset_3)
                test_sampler_4 = torch.utils.data.distributed.DistributedSampler(self.test_dataset_4)

            else:
                train_labeled_sampler = None
                train_unlabeled_sampler = None
                test_sampler = None

        except RuntimeError:
            train_labeled_sampler = None
            train_unlabeled_sampler = None
            test_sampler = None
            
        


        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.config.seed)

        train_labeled_loader = torch.utils.data.DataLoader(dataset=self.train_labeled_dataset,
                                batch_size= self.config.data.samples_per_gpu,
                                shuffle=False if train_labeled_sampler is not None else True,
                                sampler=train_labeled_sampler, num_workers= self.config.data.workers_per_gpu,
                                pin_memory=True, drop_last=True, generator = g, worker_init_fn = seed_worker)
        # train_unlabeled_loader = torch.utils.data.DataLoader(dataset=self.train_unlabel_dataset,
        #                         batch_size= self.config.data_loader['train_loader']['args']['batch_size'],
        #                         shuffle=False if train_unlabeled_sampler is not None else True,
        #                         sampler=train_unlabeled_sampler, num_workers= self.config.data_loader['train_loader']['args']['num_workers'],
        #                         pin_memory=True, drop_last=self.config.data_loader['train_loader']['args']['drop_last'])
        # train_unlabeled_reader = DataReader(train_unlabeled_loader)
        
        test_loader_0 = torch.utils.data.DataLoader(dataset=self.test_dataset_0,
                                batch_size= 1,
                                shuffle=False,
                                sampler=None, num_workers= self.config.data.workers_per_gpu,
                                pin_memory=True, drop_last=False, generator = g, worker_init_fn = seed_worker)
        
        test_loader_1 = torch.utils.data.DataLoader(dataset=self.test_dataset_1,
                                batch_size= 1,
                                shuffle=False,
                                sampler=None, num_workers= self.config.data.workers_per_gpu,
                                pin_memory=True, drop_last=False, generator = g, worker_init_fn = seed_worker)
        test_loader_2 = torch.utils.data.DataLoader(dataset=self.test_dataset_2,
                                batch_size= 1,
                                shuffle=False,
                                sampler=None, num_workers= self.config.data.workers_per_gpu,
                                pin_memory=True, drop_last=False, generator = g, worker_init_fn = seed_worker)
        test_loader_3 = torch.utils.data.DataLoader(dataset=self.test_dataset_3,
                                batch_size= 1,
                                shuffle=False,
                                sampler=None, num_workers= self.config.data.workers_per_gpu,
                                pin_memory=True, drop_last=False, generator = g, worker_init_fn = seed_worker)
        test_loader_4 = torch.utils.data.DataLoader(dataset=self.test_dataset_4,
                                batch_size= 1,
                                shuffle=False,
                                sampler=None, num_workers= self.config.data.workers_per_gpu,
                                pin_memory=True, drop_last=False, generator = g, worker_init_fn = seed_worker)




        return train_labeled_loader, None, test_loader_0, test_loader_1, test_loader_2, test_loader_3, test_loader_4