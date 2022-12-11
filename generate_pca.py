import test_inference
from test_inference import *
from timm.data import ImageDataset, create_loader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

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
    checkpoint_path = '/home/qilei/.TEMP/FDWJ/v3_2/work_dir/resnext50_32x4d-224/clean_model_best.pth'
    img_path = '/home/qilei/.TEMP/FDWJ/v3_2/test/'
    model_name = 'resnext50_32x4d'
    num_classes=2

    model = create_model(
        model_name,
        num_classes=num_classes,
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
    out_features = torch.zeros((0, 2048), dtype=torch.float32)
    targets = torch.zeros((0), dtype=torch.float32)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            out_feature = model.forward_features1(input)
            #print(out_feature.shape)
            out_features = torch.cat((out_features, out_feature.detach().cpu()), 0)
            targets = torch.cat((targets, target.detach().cpu()), 0)

    out_features = np.array(out_features)
    target_labels = np.array(targets)
    print(out_features.shape)
    print(target_labels.shape)

    tsne = TSNE(2, perplexity=50, init='pca')
    tsne_proj = tsne.fit_transform(out_features)
    tx = tsne_proj[:, 0]
    ty = tsne_proj[:, 1]    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(111)
    colors_per_class = num_classes
    cmap = cm.get_cmap('tab20')

    #classes = ['Upper Esophagus', 'Lower Esophagus',
    #                    'Upper Gastric Body', 'Middle Gastric Body', 'Lower Gastric Body',
    #                    'Gastric Antrum', 'Esophagogastric Angle', 'Duodenal Bulb', 'Descending Duodenum',
    #                    'Lower Fundus', 'Upper Fundus', 'Background']
    classes = ['0','1']
    # for every class, we'll add a scatter plot separately
    for label in range(colors_per_class):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(target_labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        # current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format
        color = np.array(cmap(label)).reshape(1, 4)
        # color = cmap[label]
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, s=10, c=color, label=classes[label])

    # build a legend using the labels we set previously
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    # finally, show the plot
    #plt.savefig(os.path.join("/home/qilei/.TEMP/gastro_position_clasification_11/work_dir/swin_base_patch4_window7_224-224", model_name+'_pca.jpg'))
    plt.savefig(os.path.join("/home/qilei/.TEMP/FDWJ/v3_2/work_dir/resnext50_32x4d-224/", model_name+'_pca.jpg'))
if __name__ == '__main__':
    generate_pca()