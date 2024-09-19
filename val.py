# train.py
#!/usr/bin/env	python3

""" valuate network using pytorch
    author Cecilia Diana-Albelda
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg_valid
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
from utils import *
import function


args = cfg_valid.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch']

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

'''segmentation data'''

val_transforms = Compose(
        [
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ]
    )
    
'''data end'''
    
if 'brats' in args.dataset: # BraTS dataset 
    '''Brain Tumor data'''
    brats_dataset = Brats(args, args.data_path, mode = 'Validation' , transform =  val_transforms)
    
    dataset_size = len(brats_dataset)
    indices = list(range(dataset_size))
    test_sampler = SubsetRandomSampler(indices[:])
    print('length test sampler: ', len(test_sampler))
    nice_test_loader = DataLoader(brats_dataset, batch_size=args.b, sampler=test_sampler,  num_workers=args.w, pin_memory=True)

'''begin evaluation'''

if args.mod: 
    net.eval()
    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')
    
