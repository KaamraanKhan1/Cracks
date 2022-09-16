#Creating a custom data loader
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
import cv2
import os
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import glob
import pickle


class CustomDataset(Dataset):
    def __init__(self,path):
        self.imgs_path =path
        self.data = []
        file_list = glob.glob(self.imgs_path + "*")
        file_list=[i.split("/")[-1] for i in file_list]

        for img_path in glob.glob(path+file_list[0] + "/*"):
          img_name=img_path.split('/')[-1]
          label_path=path+file_list[0]+"/"+img_name
          image_path=path+file_list[1]+"/"+img_name
          self.data.append([image_path, label_path])



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image_path, label_path = self.data[idx]
        img = cv2.imread(image_path)
        #Moving axis
        img= np.moveaxis(img, 2, 0)
        #Replacing nan with number
        img= np.nan_to_num(img)
        #Performing normalization
        img= (img-np.min(img))/(np.max(img)-np.min(img))
        label = cv2.imread(label_path)
        #Taking only 1 channel
        label=label[:,:,0]
        label[label==2]=0
        label=np.expand_dims(label, axis=0)

        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label)

        return img_tensor, label_tensor
