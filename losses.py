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

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True

class Loss :
    #Function to calculate dice loss
    def dice_loss(self,input,target):
      smooth = 1
      iflat = input.view(-1)
      tflat = target.view(-1)
      intersection = (iflat * tflat).sum()

      return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

    #Function to calculate BCE loss
    def BCE(self,input,target):
      loss_criteria = torch.nn.BCELoss()
      return loss_criteria(input,target)

    #Function to calculate weighted BCE loss
    def weighted_BCE(self,input,target,w):
      class_weight = torch.tensor([w],dtype=torch.float32).to(device)
      loss_criteria=nn.BCEWithLogitsLoss(pos_weight=class_weight)
      return loss_criteria(input,target)
