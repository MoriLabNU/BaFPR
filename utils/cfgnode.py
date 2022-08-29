import numpy as np
import sys
import os
import shutil
import time

# import from the top dir
import configs

def mk_dir_with_time(root, name):
    if not os.path.exists(os.path.join(root, name)):
        os.mkdir(os.path.join(root, name))
    exp_folder = name#os.path.join(root, name)

    
    dir_with_time = os.path.join(root, exp_folder, name + '_'+ str(time.time()))
    os.mkdir(dir_with_time)
    return dir_with_time



def copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns('exp','*.pth','*.png', '*.pyc', '__pycache__')):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        #import pdb; pdb.set_trace()
        if os.path.isdir(s):
            if './__pycache__' in s or './exp' in s:
                continue
            shutil.copytree(s, d, symlinks, ignore=ignore)
        else:
            if item[-2:] == 'py':
                shutil.copy2(s, d)

def mk_exp_dir(config):
    root = config.save_to
    
    name = config.exp_name

    if config.save_to is not None:
        dir_with_time = mk_dir_with_time(root, name)
        
        config.save_to = dir_with_time
        code_dir = dir_with_time + '/code'
        log_dir = dir_with_time + '/log'
        result_dir = dir_with_time + '/log/results'
        
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        # ! need to be checked
        # ! uncomment if needs to store the code version instead of using version control
        # copytree('../', code_dir)
    return config

class CfgNode(dict):
    """
    forked from DARS
    # TODO: Spawn failed to load this warpping wihtout set/get attr
    # TODO: Check source code for spawn
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def check_args(config):
    assert type(config) is dict, print('loading config file failed')
    #assert config['trainer'] is not None, 'trainer config not set'
    #assert config['distributed'] is not None, 'cannot find dist config'

def class2dict(cls):
    new_dict = {}
    for k, v in cls.__dict__.items():
        if not k.startswith('__'):
            new_dict[k] = v
    return new_dict


def load_args_from_configfile(args):
    
    dataset_name, config_name = args.config.split('.')
    args_seed = args.seed
    assert type(config_name) is str, print('pls use str to indicate the config file')

    #config = config_registry.build(config_name)
    config = getattr(getattr(configs, dataset_name), config_name)
    config = class2dict(config)

    
    
    
    check_args(config)
    config = CfgNode(config)
    config = mk_exp_dir(config)
    
    if args_seed is None:
        seed = config.get('seed', None)
    else:
        args_seed = int(args_seed)
        assert isinstance(args_seed, int), f'not allowed seed type {type(args_seed)}'
        seed = args_seed
        config['seed'] = seed
    
    # TODO: support multiple GPU training
    # if config['distributed'].get('dist_url', None) == 'env://' and \
    #     config['distributed'].get('world_size', None) == -1:
    #     config['distributed']['world_size'] = os.environ['WORLD_SIZE']
    # distributed = config['distributed'].get('multiprocessing_distributed', False) 
    # ngpus_per_node = len(config.distributed['train_gpus'])
    return config, seed#, distributed, ngpus_per_node




if __name__ == '__main__':
    # ! for debugging
    pass
    