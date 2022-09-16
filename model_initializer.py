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
import pickle
# from models import UNet,Crack_UNet

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True

class Initializer :
  def __init__(self,direc,model,LEARNING_RATE):
        self.direc =direc
        self.model=model
        self.LEARNING_RATE=LEARNING_RATE

  def initialise_model(self):
    # Checkpointing
    checkpoints = os.path.join(self.direc,"Problem")
    best_checkpoints = os.path.join(self.direc,"Best_weights")

    #If checkpoints doesnot exist, Creating a folder for checkpoints and best checkpoints
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
        os.makedirs(best_checkpoints)
    checkpoint_file = os.path.join(checkpoints, "checkpoints.pt")
    best_checkpoint_file = os.path.join(best_checkpoints, "checkpoints.pt")

    # Initialising Network
    network = self.model
    network.to(device)
    #Adam optimizer
    optimizer: optim.Adam = torch.optim.Adam(network.parameters(), lr=self.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    print("Initialised Network, Optimizer and Loss")

    train_losses, test_losses,train_accuracies,test_accuracies =[],[],[],[]
    best_score=0

    # Load checkpoints if exist
    if os.path.exists(checkpoint_file):
        print("Loading from Previous Checkpoint...")
        checkpoint = torch.load(checkpoint_file)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        network.train()

        file1 = open(self.direc+'Train_acc.pkl', 'rb')
        train_accuracies = pickle.load(file1)

        file2 = open(self.direc+'Test_acc.pkl', 'rb')
        test_accuracies = pickle.load(file2)

        file3 = open(self.direc+'Train_loss.pkl', 'rb')
        train_losses = pickle.load(file3)

        file4 = open(self.direc+'Test_loss.pkl', 'rb')
        test_losses = pickle.load(file4)

        file5 = open(self.direc+'Best_score.pkl', 'rb')
        best_score = pickle.load(file5)


    else:
        print("No previous checkpoints exist, initialising network from start...")

    return best_checkpoint_file,checkpoint_file, network, optimizer, train_losses, test_losses,train_accuracies,test_accuracies,best_score
