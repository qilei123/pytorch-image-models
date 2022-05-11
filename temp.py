
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# extract x and y coordinates representing the positions of the images on T-SNE plot


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':

    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = sqz_net(12, hid=512)
    tsne = TSNE(2, perplexity=50, init='pca')

    test_data_dir = 'dataset/gastro_position_clasification_11/test'
    transform = A.Compose([A.Resize(224, 224),
                           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ToTensorV2()])

    testset = gastro_set(data_root=test_data_dir, transform=transform)

    test_loader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.load_state_dict(torch.load('/media/chenxi/data4/gastro_disease/weight/gastro_position_SqueezeNet_wother_11/best_ep23.pth'))
    model.to(device)
    model.eval()
    test_embeddings = torch.zeros((0, 2048), dtype=torch.float32)
    test_predictions = []
    labels = []

    with torch.no_grad():

        for data in test_loader:
            images, label, path = data[0].to(device), data[1], data[2][0]
            labels.append(label)
            embeddings = model(images, phase='test', pca=True)
            test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)

    test_embeddings = np.array(test_embeddings)

    tsne_proj = tsne.fit_transform(test_embeddings)

    tx = tsne_proj[:, 0]
    ty = tsne_proj[:, 1]
    # tz = tsne_proj[:, 2]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # tz = scale_to_01_range(tz)

    # initialize a matplotlib plot
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(111)
    colors_per_class = 12
    cmap = cm.get_cmap('tab20')

    # cmap_man = [(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    #             (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    #             (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    #             (1.0, 0.4980392156862745, 0.054901960784313725),
    #             (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)]
    # print(cmap)

    if colors_per_class == 2:
        classes = ['disease_free', 'diseased']
    elif colors_per_class == 3:
        classes = ['disease_free', 'diseased_A', 'diseased_B']
    elif colors_per_class == 5:
        classes = ['cancer', 'disease_free', 'early_cancer', 'erosive', 'ulcer']\

    elif colors_per_class == 12:
        classes = ['upper_esophagus', 'lower_esophagus',
                          'upper_gastric_body', 'middle_gastric_body', 'lower_gastric_body',
                          'gastric_antrum', 'esophagoastric_angle', 'duodenal_bulb', 'descending_duodenum',
                          'fundus_low', 'fundus_high', 'other']

    # for every class, we'll add a scatter plot separately
    for label in range(colors_per_class):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

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
    plt.savefig(os.path.join("plot_folder", 'position_sqz_pca.jpg'))



class res50_net(nn.Module):
    def __init__(self, noc, hid=512):
        super(res50_net, self).__init__()

        self.Res50 = models.resnet50(pretrained=True)
        self.Res50.fc = nn.Sequential(nn.Linear(2048, hid),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(hid, noc))
        #self.Res50.fc.out_features = 2
        #
        # self.smooth = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.ReLU()
        # )
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x, labels=None, phase='train', pca=False):

        if pca:
            self.Res50.fc = Identity()


        y = self.Res50(x)
        # s = self.smooth(y)
        # y = y * s

        if phase == 'train':
            loss = self.criterion(y, labels)
            return loss

        else:
            return y