import test_inference
from test_inference import *
from timm.data import ImageDataset, create_loader

def get_single_image_feature(image,model,transform):
    if isinstance(image,str):
        img = Image.open(image).convert('RGB')
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)    

    img_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
    img_tensor = img_tensor.cuda()
    
    with torch.no_grad():
        out_feature = model.forward_features(img_tensor)
    
    out_feature = out_feature.cpu().numpy()
    return out_feature
    

def generate_pca():
    checkpoint_path = '/home/qilei/.TEMP/gastro_position_clasification_11/work_dir/swin_base_patch4_window7_224-224/clean_model_best.pth'
    img_path = '/home/qilei/.TEMP/gastro_position_clasification_11/test/'
    model_name = 'swin_base_patch4_window7_224'
    
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

    loader = create_loader(
        ImageDataset(img_path),
        input_size=config['input_size'],
        batch_size=32,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=2,
        crop_pct=1.0)
    out_features = torch.zeros((0, 1024), dtype=torch.float32)
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            out_feature = model.forward_features(input)
            out_features = torch.cat((out_features, out_feature.detach().cpu()), 0)

    print(out_features.shape)


if __name__ == '__main__':
    generate_pca()