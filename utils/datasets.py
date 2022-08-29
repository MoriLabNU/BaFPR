import os
import imageio
import torch
from torch.utils.data import Dataset
import numpy as np 
import albumentations as A
import cv2
import albumentations.pytorch

def get_trans(train = True, aug =True, *args, **kwargs):
    
    train_trans_list  = [
            
            A.Resize(352, 352),
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            #A.pytorch.transforms.ToTensorV2(),
        ]
    if not aug:
        train_trans_list = [
            
            A.Resize(352, 352),
            A.Normalize(),
        ]
    transform = A.Compose(train_trans_list)
    
    val_trans = A.Compose([
        A.Resize(352, 352),
        A.Normalize(),
        albumentations.pytorch.transforms.ToTensorV2(),
    ])

    if train:
        return transform
    else:
        return val_trans

class base_dataset(Dataset):
    # * :param: dataset: name of data
    # * :param: idx, list of img path
    # * :param: training, bool, iff the built dataset for training
    # * :param: augmentatation, do aug besides scaling and cropping
    def __init__(self, idx, config, crop_size, scale_size, training, augmentatation):
        
        self.config = config
        self.idx = idx
        
        assert isinstance(crop_size, tuple) or isinstance(crop_size, int), 'crop size only support tuple or int'
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        assert isinstance(scale_size, tuple) or isinstance(scale_size, float) or (scale_size is None), 'scale size only support tuple or float'
        self.scale_size = scale_size

        self.default_trans = None

        assert isinstance(self.idx, list), 'idx must be list'



    def __getitem__(self, index):
        pass
            
    def __len__(self):
        return len(self.idx)


    # ! from https://github.com/marinbenc/medical-polar-training/
    # ! blob/3cb206b47159efbb7d5c79a1a97fb2bec9e94e40/polar_transformations.py#L1
    @staticmethod
    def centroid(img):
        assert len(img.shape) == 2, 'moments need gray value img'
        M = cv2.moments(img)

        if M["m00"] == 0:
            return (img.shape[0] // 2, img.shape[1] // 2)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    @staticmethod
    def target_mix(img0, label0, img1, label1):
        mask = label0 > 0
        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        mix_img = img0 * mask + img1 * (1-mask)
        mix_label = label0 * mask[...,0] + label1 * (1-mask[...,0])
        return mix_img, mix_label

    @staticmethod
    def background_mix(img0, label0, img1, label1):
        assert img0.shape == img1.shape
        assert label0.shape == label1.shape
        label0 = label0 > 0
        label1 = label1> 0
        mask = np.logical_or(label0, label1).astype(np.uint8)
        label0 = label0.astype(np.uint8)
        label1 = label1.astype(np.uint8)

        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        mix_img = img0 * mask + img1 * (1-mask)
        mix_label = label0 * mask[...,0] + label1 * (1-mask[...,0])
        return mix_img, mix_label

    
    def polor_trans(self, img, center):
        #center = self.centroid(img)
        img = img.astype(np.float32)
        value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
        polar_image = cv2.linearPolar(img, center, value, cv2.WARP_FILL_OUTLIERS)
        #polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return polar_image
        

    def inverse_polar_trans(self, input_img, center):
        input_img = input_img.astype(np.float32)
        #input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
        value = np.sqrt(((input_img.shape[1]/2.0)**2.0)+((input_img.shape[0]/2.0)**2.0))
        polar_image = cv2.linearPolar(input_img, center, value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        inv_polar_image = polar_image
        return inv_polar_image




class BuildDataset(base_dataset):
    # * :param: dataset: name of data
    # * :param: idx, list of img path
    # * :param: training, bool, iff the built dataset for training
    # * :param: augmentatation, do aug besides scaling and cropping
    def __init__(self, idx, config, crop_size, scale_size, training, augmentatation):
        super().__init__( idx, config, crop_size, scale_size, training, augmentatation)

        
        self.default_trans = get_trans(crop_size=crop_size, scale_size=scale_size, train=training, aug=augmentatation)
        self.to_tensor =  A.pytorch.transforms.ToTensorV2()
        self.training = training


        assert isinstance(self.idx, list), 'idx must be list'

        # # * color space aug
        # self.color1, self.color2 = [], []
        # #import pdb; pdb.set_trace()
        # for name in self.idx:
        #     if name.split('/')[-1][:-4].isdigit():
        #         self.color1.append(name)
        #     else:
        #         self.color2.append(name)



    def __getitem__(self, index):
        image_dict ={}
        image = cv2.imread(self.idx[index])
        if not self.training:
            image_dict['origin'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_root = self.idx[index].replace('images', 'masks')
        
        label = cv2.imread(label_root, cv2.IMREAD_GRAYSCALE)
        label = label > 0
        if len(label.shape) == 3:
            raise AssertionError()

        # if np.random.uniform() < 0.3 and self.training and self.config.get('background_mix', False):
        #     idx_sub = np.random.randint(0, len(self.idx[index]))
        #     image_sub = cv2.imread(self.idx[idx_sub])
        #     image_sub = cv2.cvtColor(image_sub, cv2.COLOR_BGR2RGB)
        #     label_root_sub = self.idx[idx_sub].replace('image', 'mask')
        #     label_sub = cv2.imread(label_root_sub, cv2.IMREAD_GRAYSCALE)/255.0
        #     if len(label_sub.shape) == 3:
        #         raise AssertionError()

        #     image_sub = cv2.resize(image_sub, (image.shape[1],  image.shape[0]))
        #     label_sub = cv2.resize(label_sub, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
        #     image, label = self.background_mix(image, label, image_sub, label_sub)

        
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        augmented = self.default_trans(image=image, mask=label)
        aug_image, aug_label = augmented['image'], augmented['mask']

        
        
        
        if self.training:
            cart_aug = self.to_tensor(image=aug_image, mask=aug_label)
            image_dict['aug_label'] = cart_aug['mask']
            image_dict['aug_image'] = cart_aug['image']
        else:
            image_dict['aug_label'] = aug_label
            image_dict['aug_image'] = aug_image

        if self.config.get('enable_polar_consist', False) and self.training:
            
            cX, cY = self.centroid(aug_label)
            
                
            if self.config.get('random_center', False):
                if np.random.uniform() < 0.3:
                    
                    center_max_shift = 0.05 * self.config.crop_size[0]
                    cX = cX + np.random.uniform(-center_max_shift, center_max_shift)
                    cY = cY + np.random.uniform(-center_max_shift, center_max_shift)
            aug_label_polor = self.polor_trans(aug_label, center = (cX, cY))
            aug_image_polar = self.polor_trans(aug_image, center = (cX, cY))

            image_dict['polar_center'] = (cX, cY)
            polor_aug = self.to_tensor(image=aug_image_polar, mask = aug_label_polor)
            image_dict['aug_image_polar'] = polor_aug['image']
            image_dict['aug_label_polor'] = polor_aug['mask']

        return image_dict, self.idx[index]