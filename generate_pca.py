import test_inference
from test_inference import *

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
    img_path = '/home/qilei/.TEMP/gastro_position_clasification_11/test/0/20191015_1601_1610_w_779.jpg'
    model_name = 'swin_base_patch4_window7_224'
    
    model, transform = init_model(model_name,checkpoint_path)

    image = cv2.imread(img_path)

    out_feature = get_single_image_feature(image,model,transform)

    print(out_feature)


if __name__ == '__main__':
    generate_pca()