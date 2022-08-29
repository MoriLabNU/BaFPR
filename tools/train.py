import argparse
import copy
import socket
import os
import os.path as osp
import time
import warnings
import random
import numpy as np 
import sys 
sys.path.append(os.getcwd())

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utils import load_config, _get_logger

from trainer_UACANet import Trainer as UACA_Trainer
from trainer import Trainer

import pprint



def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def main_worker(config):

    


    # * recording exp when necessary
    global writer, img_writer, logger
    if config.get('save_to', None) is not None:
        writer, img_writer, logger = _get_logger(os.path.join(config.save_to, 'log'))
        logger.info(config)
    else:
        pprint.pprint(config)
        writer, img_writer, logger = None, None, None
    assert torch.cuda.is_available(), f'not supporting cpu training'
    
    # torch.cuda.set_device(0)

    # print('world size')
    # print(dist.get_world_size())
    # exit()
    
    
    if config.model.backbone =='UACA_L':
        trainer_ins = UACA_Trainer(config, logger_tuple = (writer, img_writer, logger))
        trainer_ins.train()
    else:
        trainer_ins = Trainer(config, logger_tuple = (writer, img_writer, logger))
        trainer_ins.train()


def main(args):

    # * load config file to args
    assert args.config, 'pls specify the config file for training'
    config, seed  = load_config(args)
    print(f'using seed {seed}')
    if seed is not None:
        
        print(f'---------------------------------Fixing Random Seed--------------------------')
        # ! Pls note that even though random seed is fixed, the result could be diff when using NCCL
        # ! backend: https://github.com/pytorch/pytorch/issues/48576, https://github.com/cybertronai/imagenet18/blob/dev/train.py
        # ! we believe this problem can be sloved by setting NCCL ring order. \
        # ! However, we didn't fix the order for computational effiency during our exps.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    if args.world_size > 1:
        #print('init')
        #port = find_free_port()
        #print(port)
        #master_address = f"tcp://127.0.0.1:{port}"
        #print(master_address)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        
        

    
    main_worker(config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--seed', default=None)
    args = parser.parse_args()
    main(args)