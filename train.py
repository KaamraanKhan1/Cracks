
#Importing LIbraries
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
import warnings
from losses import Loss
warnings.filterwarnings("ignore")
os.system('CUDA_LAUNCH_BLOCKING=1')

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True


class Train:
  def __init__(self,direc,NUM_EPOCHS,LEARNING_RATE,w_d,w_b,w_wb,weight):
        self.direc =direc
        self.LEARNING_RATE=LEARNING_RATE
        self.NUM_EPOCHS=NUM_EPOCHS
        self.w_d=w_d
        self.w_b=w_b
        self.w_wb=w_wb
        self.weight=weight

  def train_model(self,best_checkpoint_file,checkpoint_file,  network, optimizer, test_loader, train_loader,train_losses, test_losses,train_accuracies,test_accuracies,best_score ):
        loss_=Loss()
        for epoch in range(self.NUM_EPOCHS):
            running_corrects_train = 0
            running_loss_train = 0.0

            count=0
            c=0
            Confusion=[[0, 0],[0 ,0 ]]
            Test_Confusion=[[0, 0],[0 ,0 ]]

            for input_, labels in train_loader:
                if torch.cuda.is_available():
                    input_ = input_.to(device)
                    labels = labels.to(device)
                output = network(input_.float())

                #Calculating dice,bce loss and weighted loss
                loss_d=loss_.dice_loss(output.float(), labels.float())
                loss_b = loss_.BCE(output.float(), labels.float())
                loss_w_b = loss_.weighted_BCE(output.float(), labels.float(),self.weight)
                #Calculating final loss
                loss= self.w_d*loss_d+self.w_b*loss_b+self.w_wb*loss_w_b

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                output = output.view( -1)
                label_endo = labels.reshape(len(output))
                output = output.cpu()
                output = output.detach().numpy()


                label_endo = label_endo.cpu()
                label_endo = label_endo.detach().numpy()


                output[output>=0.5]=1  #Applying thresholding
                output[output<0.5]=0
                cm = confusion_matrix(label_endo,output)
                Confusion=Confusion+cm   #Storing Confusion Matrix


                running_loss_train += loss.item() * input_.size(0)
                count+=input_.size(0)
                if count==400:
                  break

            #Calculating and storing dice raio, accuracy, confusion matrix and losses
            FP = Confusion.sum(axis=0) - np.diag(Confusion)
            FN = Confusion.sum(axis=1) - np.diag(Confusion)
            TP = np.diag(Confusion)
            DR = 2*TP / (2*TP + FP + FN)
            train_acc = (Confusion[0][0]+Confusion[1][1])/(Confusion[0][0]+Confusion[1][1]+Confusion[0][1]+Confusion[1][0])

            train_loss = running_loss_train / count
            train_accuracies.append(DR[1])
            train_losses.append(train_loss)  # loss computed as the average on mini-batches


            print("______________________________________________________________________________")
            print("epoch:",len(train_accuracies), "Train loss : ",train_loss)
            print("-Train accuracy:", train_acc, flush=True)
            print("Dice Ratio : ",DR)
            print(Confusion)
            print()


            running_corrects_test = 0
            running_loss_test = 0.0

            #Performing same thing for validation data
            with torch.set_grad_enabled(False):
                optimizer.zero_grad()
                for test_input, test_labels in test_loader:
                    if torch.cuda.is_available():
                        test_input = test_input.to(device)
                        test_labels = test_labels.to(device)
                    test_output = network(test_input.float())

                    output = test_output.view( -1)
                    label_endo = test_labels.reshape(len(output))
                    output = output.cpu()
                    output = output.detach().numpy()
                    label_endo = label_endo.cpu()
                    label_endo = label_endo.detach().numpy()

                    output[output>=0.5]=1
                    output[output<0.5]=0
                    cm = confusion_matrix(label_endo,output)
                    Test_Confusion=Test_Confusion+cm

                    test_loss_d=loss_.dice_loss(test_output.float(), test_labels.float())
                    test_loss_b = loss_.BCE(test_output.float(), test_labels.float())
                    test_loss_w_b = loss_.weighted_BCE(test_output.float(), test_labels.float(),self.weight)

                    test_loss= self.w_d*test_loss_d+self.w_b*test_loss_b+self.w_wb*test_loss_w_b
                    running_loss_test += test_loss.item() * test_input.size(0)
                    c+=test_input.size(0)

                FP = Test_Confusion.sum(axis=0) - np.diag(Test_Confusion)
                FN = Test_Confusion.sum(axis=1) - np.diag(Test_Confusion)
                TP = np.diag(Test_Confusion)
                DR = 2*TP / (2*TP + FP + FN)
                test_acc = (Test_Confusion[0][0]+Test_Confusion[1][1])/(Test_Confusion[0][0]+Test_Confusion[1][1]+Test_Confusion[0][1]+Test_Confusion[1][0])
                test_loss = running_loss_test / c
                test_accuracies.append(DR[1])
                test_losses.append(test_loss)

                print("Validation Accuracy:", test_acc, flush=True)
                print("Dice Ratio : ",DR)
                print(Test_Confusion)

            #Saving the best weights
            if Test_Confusion[1][1]>best_score:
              best_score=Test_Confusion[1][1]
              torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, best_checkpoint_file)

              with open(self.direc+'Best_score.pkl', 'wb') as f:
                  pickle.dump(best_score, f)


            #After every 5 epochs, stroing the weights, dice ratio and losses
            if epoch % 5 == 0:
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_file)

                with open(self.direc+'Train_acc.pkl', 'wb') as f:
                  pickle.dump(train_accuracies, f)

                with open(self.direc+'Test_acc.pkl', 'wb') as f:
                    pickle.dump(test_accuracies, f)

                with open(self.direc+'Train_loss.pkl', 'wb') as f:
                    pickle.dump(train_losses, f)

                with open(self.direc+'Test_loss.pkl', 'wb') as f:
                    pickle.dump(test_losses, f)


        return train_losses, test_losses,train_accuracies,test_accuracies




# train_losses, test_losses,train_accuracies,test_accuracies= initialise_model(trainloader, valloader)
