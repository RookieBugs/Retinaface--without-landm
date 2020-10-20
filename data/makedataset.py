#-*- coding: utf-8 -*-
import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                self.words.append(labels_copy)
                labels.clear()

            path = line[:]
            path = txt_path.replace('label.txt','train_img/') + path
            self.imgs_path.append(path)

            txt_line = path.replace('img','txt')
            txt_line = txt_line.replace('.jpg','.txt')
            try:
                f = open(txt_line,'r')
            except:
                self.imgs_path.pop()
                continue
            f_lines = f.readlines()
            for f_line in f_lines:
                f_line = f_line.split()
                label = [float(x) for x in f_line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 5))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))
            # bbox
            annotation[0, 0] = width*(label[1]-label[3]/2)  # x1
            annotation[0, 1] = height*(label[2]-label[4]/2)  # y1
            annotation[0, 2] = width*(label[1] + label[3]/2)  # x2
            annotation[0, 3] = height*(label[2] + label[4]/2)  # y2

            # # landmarks
            # annotation[0, 4] = 0.0    # l0_x
            # annotation[0, 5] = 0.0    # l0_y
            # annotation[0, 6] = 0.0    # l1_x
            # annotation[0, 7] = 0.0   # l1_y
            # annotation[0, 8] = 0.0   # l2_x
            # annotation[0, 9] = 0.0   # l2_y
            # annotation[0, 10] = 0.0  # l3_x
            # annotation[0, 11] = 0.0  # l3_y
            # annotation[0, 12] = 0.0  # l4_x
            # annotation[0, 13] = 0.0  # l4_y
            # if (annotation[0, 4]<0):
            #     annotation[0, 14] = -1
            # else:
            annotation[0, 4] = 1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target