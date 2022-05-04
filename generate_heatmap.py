import torch
import timm
m = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
o = m(torch.randn(2, 3, 224, 224))
print(f'Unpooled shape: {o.shape}')