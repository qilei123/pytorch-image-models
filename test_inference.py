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
from PIL import Image
import cv2

from timm.models import create_model, apply_test_time_pool
from timm.data import resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from timm.data.transforms_factory import create_transform

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

def init_model(model_name,checkpoint_path):
    model = create_model(
        model_name,
        num_classes=12,
        in_chans=3,
        pretrained=False,
        checkpoint_path=checkpoint_path)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    model = model.cuda()
    model.eval()
    return model,transform    

def inference_single_image(image,model,transform):
    if isinstance(image,str):
        img = Image.open(image).convert('RGB')
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)    

    img_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
    img_tensor = img_tensor.cuda()
    
    with torch.no_grad():
        out = model(img_tensor)
    topk = out.topk(1)[1]
    result_label = topk.cpu().numpy()[0][0]

    return result_label

def test_inference():

    checkpoint_path = '/home/qilei/.TEMP/gastro_position_clasification_11/work_dir/swin_base_patch4_window7_224-224/model_best.pth.tar'
    img_path = '/home/qilei/.TEMP/gastro_position_clasification_11/test/0/20191015_1601_1610_w_779.jpg'
    model_name = 'swin_base_patch4_window7_224'
    
    model, transform = init_model(model_name,checkpoint_path)

    img_path = cv2.imread(img_path)

    result_label = inference_single_image(img_path,model,transform)

    print(result_label)

if __name__ == '__main__':
    test_inference()