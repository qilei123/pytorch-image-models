#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

def test_inference():
    model = create_model(
        'vit_base_patch32_224',
        num_classes=12,
        in_chans=3,
        pretrained=False,
        checkpoint_path='/home/qilei/.TEMP/gastro_position_clasification_11/work_dir/vit_base_patch32_224-224/model_best.pth.tar')

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    model = model.cuda()
    model.eval()
    
    image = cv2.imread()

if __name__ == '__main__':
    test_inference()