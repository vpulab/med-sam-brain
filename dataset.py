""" train and test dataset

author Cecilia Diana-Albelda
"""
import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset
import SimpleITK as sitk
from einops import rearrange

from utils import random_click



class Brats(Dataset):
    def __init__(self, args, data_path , transform = None,  mode = 'Training',prompt = 'click', plane = False):
        self.data_path = data_path
        self.folder = 'brats_ssa' if args.dataset == 'brats_ssa' else ('brats_2020' if args.dataset == 'brats_2020' else ('brats_men' if args.dataset == 'brats_men' else ('brats_ped' if args.dataset == 'brats_ped' else 'brats')))
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, self.folder)) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.four_chan = args.four_chan
        self.mri = args.mri

        self.transform = transform

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        point_label = 1
        # Get the images 
        subfolder = self.subfolders[index]
        # name = subfolder #example: BraTS-GLI-00000-000 
        
        if self.folder == 'brats_2020':
            # raw image and mask paths 
            t1_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+ '_t1.nii.gz') 
            t1c_path = os.path.join( subfolder + '/'+ subfolder.split('/')[-1]+  '_t1ce.nii.gz')
            t2_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+  '_t2.nii.gz')
            t2f_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+  '_flair.nii.gz')
            mask_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+  '_seg.nii.gz')
        
        else:
            # raw image and mask paths 
            t1_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+ '-t1n.nii.gz') 
            t1c_path = os.path.join( subfolder + '/'+ subfolder.split('/')[-1]+  '-t1c.nii.gz')
            t2_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+  '-t2w.nii.gz')
            t2f_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+  '-t2f.nii.gz')
            mask_path = os.path.join(subfolder + '/'+ subfolder.split('/')[-1]+  '-seg.nii.gz')

        # raw image and mask
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_path))) # shape:  (155, 240, 240)
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(str(t1c_path)))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(str(t2_path)))
        t2f = sitk.GetArrayFromImage(sitk.ReadImage(str(t2f_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) # shape:  (155, 240, 240)

        # Change mask: enhanching/non-enh,necrosis/edema -> whole-tumor 
        if mask is not None:
            # original = mask
            # 1: NCR - 2: ED - 3: ET 
            wt = mask != 0
            mask = np.zeros_like(mask) 
            mask[wt] = 1 

        # first click is the target agreement among most raters
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        
        # We construct an image in which the 4 MRI modalities are represented
        if self.four_chan == True:
            img =  np.stack([t1,t1c,t2,t2f]) #shape: (4, 155, 240, 240) 
            img = rearrange(img, 'c d h w -> c h w d') #shape: (4, 240, 240, 155)
            mask = np.expand_dims(mask, axis=0) #shape: (1, 155, 240, 240)
            mask = rearrange(mask, 'c d h w -> c h w d') #shape: (1, 240, 240, 155)
        else:
            if self.mri == 't1':
                img =  np.stack([t1, t1, t1]) #shape: (3, 155, 240, 240) 
                img = rearrange(img, 'c d h w -> c h w d') #shape: (3, 240, 240, 155)
                mask = np.expand_dims(mask, axis=0) #shape: (1, 155, 240, 240)
                mask = rearrange(mask, 'c d h w -> c h w d') #shape: (1, 240, 240, 155)
            elif self.mri == 't1c':
                img =  np.stack([t1c, t1c, t1c]) #shape: (3, 155, 240, 240) 
                img = rearrange(img, 'c d h w -> c h w d') #shape: (3, 240, 240, 155)
                mask = np.expand_dims(mask, axis=0) #shape: (1, 155, 240, 240)
                mask = rearrange(mask, 'c d h w -> c h w d') #shape: (1, 240, 240, 155)
            elif self.mri == 't2':
                img =  np.stack([t2, t2, t2]) #shape: (3, 155, 240, 240) 
                img = rearrange(img, 'c d h w -> c h w d') #shape: (3, 240, 240, 155)
                mask = np.expand_dims(mask, axis=0) #shape: (1, 155, 240, 240)
                mask = rearrange(mask, 'c d h w -> c h w d') #shape: (1, 240, 240, 155)
            elif self.mri == 't2f':
                img =  np.stack([t2f, t2f, t2f]) #shape: (3, 155, 240, 240) 
                img = rearrange(img, 'c d h w -> c h w d') #shape: (3, 240, 240, 155)
                mask = np.expand_dims(mask, axis=0) #shape: (1, 155, 240, 240)
                mask = rearrange(mask, 'c d h w -> c h w d') #shape: (1, 240, 240, 155)
            else: 
                img =  np.stack([t1, t2, t2f]) #shape: (3, 155, 240, 240) 
                img = rearrange(img, 'c d h w -> c h w d') #shape: (3, 240, 240, 155)
                mask = np.expand_dims(mask, axis=0) #shape: (1, 155, 240, 240)
                mask = rearrange(mask, 'c d h w -> c h w d') #shape: (1, 240, 240, 155)

        image_meta_dict = {'filename_or_obj':subfolder} # subfolders: patient-id 

        data = {
                'image':img,
                'label': mask,
                'p_label':point_label,
              #  'pt':pt,
                'image_meta_dict':image_meta_dict,
            }

        if self.transform:
            state = torch.get_rng_state()
            transformed = self.transform(data)
            img = transformed['image'] 
            mask = transformed['label'] 
            torch.set_rng_state(state)

            data = {
                    'image':img,
                    'label': mask,
                    'p_label':point_label,
                #  'pt':pt,
                    'image_meta_dict':image_meta_dict,
                }
        
        else:
            state = torch.get_rng_state()
            data = {
                    'image':img,
                    'label': mask,
                    'p_label':point_label,
                #  'pt':pt,
                    'image_meta_dict':image_meta_dict,
                }
            torch.set_rng_state(state)

        return data
