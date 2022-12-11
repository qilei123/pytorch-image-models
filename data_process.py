import os
import sys
import cv2
import numpy as np
from collections import Counter
from PIL import Image

def merge_adenoma_records():

    record_file1 = open("/data3/qilei_chen/DATA/polyp_xinzi/preprocessed_4_classification/train.txt")

    record_file2 = open("/data3/qilei_chen/DATA/polyp_xinzi/D2/preprocessed/train.txt")

    record_file3 = open("/data3/qilei_chen/DATA/polyp_xinzi/D1_D2/train.txt","w")

    record = record_file1.readline()

    while record:

        record_file3.write("D1_train/"+record)

        record = record_file1.readline()

    record = record_file2.readline()

    while record:

        record_file3.write("D2_train/"+record)

        record = record_file2.readline()

def crop_img(img,roi=None):
    if roi==None:
        if isinstance(img,str):
            img = cv2.imread(img)
        arr = np.asarray(img)
        combined_arr = arr.sum(axis=-1) / (255 * 3)
        truth_map = np.logical_or(combined_arr < 0.07, combined_arr > 0.95)
        threshold = 0.6
        y_bands = np.sum(truth_map, axis=1) / truth_map.shape[1]
        top_crop_index = np.argmax(y_bands < threshold)
        bottom_crop_index = y_bands.shape[0] - np.argmax(y_bands[::-1] < threshold)

        truth_map = truth_map[top_crop_index:bottom_crop_index, :]

        x_bands = np.sum(truth_map, axis=0) / truth_map.shape[0]
        left_crop_index = np.argmax(x_bands < threshold)
        right_crop_index = x_bands.shape[0] - np.argmax(x_bands[::-1] < threshold)

        cropped_arr = arr[top_crop_index:bottom_crop_index, left_crop_index:right_crop_index, :]
        roi = [left_crop_index,top_crop_index, right_crop_index,bottom_crop_index]
        toolbar_end = cropped_arr.shape[0]
        for i in range(cropped_arr.shape[0] - 1, 0, -1):
            c = Counter([tuple(l) for l in cropped_arr[i, :, :].tolist()])
            ratio = c.most_common(1)[0][-1] / cropped_arr.shape[1]
            if ratio < 0.3:
                toolbar_end = i
                break

        cropped_arr = cropped_arr[:toolbar_end, :, :]
        return cropped_arr,roi
    else:
        if isinstance(img,str):
            img = Image.open(img)
        arr = np.asarray(img)
        #print(roi)
        #print(arr[roi[1]:roi[3],roi[0]:roi[2],:])
        return arr[roi[1]:roi[3],roi[0]:roi[2],:]
    #return Image.fromarray(cropped_arr)

def crop_FD():
    roi = [665, 37, 1821, 1043]

    root_dir = '/home/qilei/.TEMP/放大胃镜图片筛选/'

    record_names = ['v3_test','v3_train','v1_test','v1_train']

    for record_name in record_names:
        print(record_name)
        records = open(root_dir+record_name+'.txt',encoding="utf-8")

        record = records.readline()

        while record:
            record = record.split(' ')
            record[0] = record[0].replace("\\",'/')
            if os.path.exists(os.path.join(root_dir,record[0])):

                image = cv2.imdecode(np.fromfile(os.path.join(root_dir,record[0]), dtype=np.uint8), -1)

                image,_ = crop_img(image)

                save_dir  = os.path.join(root_dir, record_name, str(record[1][:-1]))

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cv2.imwrite(os.path.join(save_dir,os.path.basename(record[0])),image)
            else:
                print(os.path.join(root_dir,record[0]))

            record = records.readline()

def single_crop_FD(img_dir,save_dir):

    image = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)

    image, _ = crop_img(image)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(os.path.join(save_dir, os.path.basename(img_dir)), image)

def checkAnormalROI(roi):
    ratio =  (roi[2]-roi[0])/(roi[3]-roi[1])

    if ratio>1.25 or ratio<1:
        return True
    return False

def crop_FD_v2():
    roi = [665, 37, 1821, 1043]

    root_dir = '/home/qilei/.TEMP/放大胃镜图片筛选/'

    abnormal_save_dir = '/home/qilei/.TEMP/放大胃镜图片筛选/abnormal_roi_images/'

    record_names = ['v3_test','v3_train',]#'v1_test','v1_train']

    for record_name in record_names:
        print(record_name)
        records = open(root_dir+record_name+'.txt',encoding="utf-8")

        record = records.readline()

        while record:
            record = record.split(' ')
            record[0] = record[0].replace("\\",'/')
            if os.path.exists(os.path.join(root_dir,record[0])):

                image = cv2.imdecode(np.fromfile(os.path.join(root_dir,record[0]), dtype=np.uint8), -1)

                crop_image,roi = crop_img(image)

                if checkAnormalROI(roi):
                    if os.path.exists(os.path.join(abnormal_save_dir,'crop2',os.path.basename(record[0]))):
                        crop_image = cv2.imdecode(np.fromfile(os.path.join(abnormal_save_dir,'crop2',os.path.basename(record[0])), dtype=np.uint8), -1)

                        save_dir = os.path.join(root_dir, record_name, str(record[1][:-1]))

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        cv2.imwrite(os.path.join(save_dir, os.path.basename(record[0])), crop_image)
                    else:
                        print(os.path.basename(record[0]))
                        cv2.imwrite(os.path.join(abnormal_save_dir, 'org', os.path.basename(record[0])), image)
                        cv2.imwrite(os.path.join(abnormal_save_dir,'crop',os.path.basename(record[0])),crop_image)
                else:
                    save_dir = os.path.join(root_dir, record_name, str(record[1][:-1]))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    cv2.imwrite(os.path.join(save_dir, os.path.basename(record[0])), crop_image)

            else:
                print(os.path.join(root_dir,record[0]))

            record = records.readline()

if __name__ == "__main__":
    #merge_adenoma_records()
    crop_FD_v2()
    pass