from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import time
import glob


class AdeClassifier(object):
    INPUT_SIZE = 224
    NUM_CLASSES = 2

    def __init__(self, model_path):
        """ Adenomatous classifier initialization
        """
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([AdeClassifier.INPUT_SIZE, AdeClassifier.INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.__model_path = model_path
        self.__model = self._init_model()

    def _init_model(self):
        """ initialize resnext model
        """
        model = models.resnext50_32x4d()
        # if the version of torchvision is not older than 0.6, please use model = models.inception_v3(transform_input=True,init_weights=False) instead otherwise downgrade scipy to 1.3.1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, AdeClassifier.NUM_CLASSES)
        model.load_state_dict(torch.load(self.__model_path))
        model.to(self.__device)
        model.eval()
        return model

    def predict(self, image: np.ndarray):
        """ classification module
         Args:
             image (np.ndarray): RGB image nd-array
         Returns:
             label (int): label of image, 0 for Adenomatous, 1 for Non Adenomatous
         """
        start_time = time.perf_counter()
        img = self.__transform(image).unsqueeze(0).to(self.__device)
        #print('preprocess ->>>>>{}'.format(time.perf_counter() - start_time))
        with torch.no_grad():
            prediction = self.__model(img)
            #print('classify ->>>>>{}'.format(time.perf_counter() - start_time))
            return torch.max(prediction, 1)[1].cpu().numpy().sum()


if __name__ == '__main__':
    model_path = "/data3/qilei_chen/DATA/polyp_xinzi/Ade_para.pkl"

    img_paths = glob.glob("/data3/qilei_chen/DATA/polyp_xinzi/D2/test/none_adenoma/*.jpg")

    count=0

    for img_path in img_paths:
        cc = AdeClassifier(model_path)
        pil_img = Image.open(img_path).convert('RGB')
        pil_img = np.asarray(pil_img)
        label = cc.predict(pil_img)
        count += label
        print(count)
        # time used to each step
        # preprocess ->>>>>0.0023029410003800876
        # classify ->>>>>0.015841885000554612
    
    print(count)