# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Cecilia Diana-Albelda
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils import tensorboard
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
from models.common import loralib as lora

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1) #learning rate decay
	

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name) # set_log_dir from utils.py
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''

train_transforms = Compose(
        [
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            
        ]
    )

if 'brats' in args.dataset: # BraTS dataset 
    '''Brain Tumor data'''
    brats_train_dataset = Brats(args, args.data_path, mode = 'Training' , transform = train_transforms)
    brats_test_dataset = Brats(args, args.data_path, mode = 'Validation' , transform =  train_transforms)

    dataset_size = len(brats_train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[split:])
    test_sampler = SubsetRandomSampler(indices[:split])

    nice_train_loader = DataLoader(brats_train_dataset, batch_size=args.b, sampler = train_sampler,  num_workers=6, pin_memory=True)
    nice_test_loader = DataLoader(brats_test_dataset, batch_size=args.b, sampler=test_sampler,  num_workers=6, pin_memory=True)
    
    '''end'''


'''checkpoint path and tensorboard'''
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = tensorboard.writer.SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, args.path_helper['log_path'].split('/')[1]))
#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
best_dice = 0.0
best_loss = 10000.0

for epoch in range(settings.EPOCH):
    net.train()
    time_start = time.time()
    loss, current_lr = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
    logger.info(f'Train loss: {loss} || @ epoch {epoch} || @ lr {current_lr}.')
    writer.add_scalar("Train_Loss", loss, epoch)

    if loss < best_loss:
        print('SAVING CHECKPOINT! - BEST LOSS')
        best_loss = loss
        is_best = True

        save_checkpoint({
        'epoch': epoch + 1,
        'model': args.net,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
       # 'best_tol': tol,
        'path_helper': args.path_helper,
    }, is_best, args.path_helper['ckpt_path'], filename="best_loss")
    else:
        is_best = False

          
    net.eval()
    if epoch: 
        tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        writer.add_scalar("Dice_Valid", edice, epoch)

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.state_dict()

        if edice > best_dice:
            print('SAVING CHECKPOINT! - BEST DICE')
            best_dice = edice
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'best_tol': tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="best_dice")
        else:
            is_best = False

writer.close()
