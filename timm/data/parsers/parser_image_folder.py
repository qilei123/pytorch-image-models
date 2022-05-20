""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os

from torch.utils import data
import csv
from timm.utils.misc import natural_key
import random

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS
from pycocotools.coco import COCO

def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ParserImageFolder(Parser):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        random.shuffle(self.samples)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

class ParserDBCSV(Parser):

    csv_anns = {"train":"trainLabels.csv","validation":"retinopathy_solution.csv"}

    def __init__(
            self,
            root,
            class_map='',
            DBbinary = False):
        super().__init__()
        self.root = root

        if self.root[-1] =='/':
            self.root = self.root[:-1]

        data_set = os.path.basename(self.root)

        self.realroot=self.root[:-len(data_set)]

        csv_ann = self.csv_anns[data_set]
        
        csvreader = csv.reader(open(os.path.join(self.realroot,csv_ann)))

        #skip fields
        next(csvreader)

        self.samples=[]

        self.class_to_idx={}

        for row_ in csvreader:
            row = []
            for i in range(2):

                row.append(row_[i])
            row[1] = float(row[1])
            if DBbinary:
                if row[1]>1:
                    row[1] = 0
                else:
                    row[1] = 1

            row[0] = os.path.join(self.root,row[0]+".jpeg")

            self.samples.append(row)

            if not row[1] in self.class_to_idx:
                self.class_to_idx[row[1]] = row[1]
        random.shuffle(self.samples)
        '''
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        '''
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

class ParserAdenoma(Parser):

    txt_anns = {"train":"train.txt","test":"test.txt"}

    def __init__(
            self,
            root,
            split='train',
            class_map='',
            DBbinary = False):
        super().__init__()

        

        self.root = root

        txt_ann = self.txt_anns[split]
        if self.root.endswith(split):
            txt_reader = open(self.root+'.txt')
        else:
            txt_reader = open(os.path.join(self.root,txt_ann))

        self.samples=[]

        row_ = txt_reader.readline()

        while row_:
            row_ = row_.replace('\n','')
            row = []
            row.append(row_[:-2])
            row.append(float(row_[-1]))    
            #if row[0] == os.path.basename(row[0]):
            #    row[0] = os.path.join(split,row[0])

            #row[1] = float(row[1])

            if DBbinary:
                if row[1]>1:
                    row[1] = 0
                else:
                    row[1] = 1

            row[0] = os.path.join(self.root,row[0])

            self.samples.append(row)
            
            row_ = txt_reader.readline()
        random.shuffle(self.samples)

        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


class ParserDental(Parser):
    coco_anns = {"train": "train_1_3.json", "test": "test_1_3.json",
                 "train_crop": "train_1_3_crop.json", "test_crop": "test_1_3_crop.json"}

    def __init__(
            self,
            root,
            split='train',
            class_map='',
            DBbinary=False):
        super().__init__()

        self.root = root

        json_ann = self.coco_anns[split]

        self.coco = COCO(os.path.join(self.root,"annotations", json_ann))

        self.samples = []

        for i in self.coco.anns:
            row = []
            row.append(self.coco.anns[i]['id'])
            #category id map to target 这里分2类，normal,abnormal
            if self.coco.anns[i]['category_id']==1:
                row.append(0)
            else:
                row.append(1)
            row.append(self.coco.anns[i]['bbox'])
            self.samples.append(row)

        random.shuffle(self.samples)

        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        else:
            pass
            #print(len(self.samples))
    def __getitem__(self, index):
        ann_id, target,bbox = self.samples[index]
        file_name = self.coco.imgs[self.coco.anns[ann_id]["image_id"]]["file_name"]
        path = os.path.join(self.root,"images",file_name)
        return open(path, 'rb'), target, bbox

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        ann_id = self.samples[index][0]
        filename = self.coco.imgs[self.coco.anns[ann_id]["image_id"]]["file_name"]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename