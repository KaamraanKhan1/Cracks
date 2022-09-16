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
warnings.filterwarnings("ignore")
os.system('CUDA_LAUNCH_BLOCKING=1')

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True



#Crack-Unet Architecture

class Crack_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(Crack_UNet, self).__init__()



        features = init_features
        self.encoder1 = Crack_UNet.crack_unet_block(in_channels, features, name="enc1")          #1024,64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = Crack_UNet.crack_unet_block(features, features * 2, name="enc2")         #512,128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = Crack_UNet.crack_unet_block(features * 2, features * 4, name="enc3")    #256,256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = Crack_UNet.crack_unet_block(features * 4, features * 8, name="enc4")    #128,512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Crack_UNet.crack_unet_block(features * 8, features * 16, name="bottleneck") #64,1024

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)  #128, 512
        self.decoder4 = Crack_UNet.crack_unet_block((features * 8) * 2, features * 8, name="dec4")            #128,512

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2 )   #256, 256
        self.decoder3 = Crack_UNet.crack_unet_block((features * 4) * 2, features * 4, name="dec3")            #256, 256

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2 )  #512,128
        self.decoder2 = Crack_UNet.crack_unet_block((features * 2) * 2, features * 2, name="dec2")           #512,128

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)    #1024,64
        self.decoder1 = Crack_UNet.crack_unet_block(features * 2, features, name="dec1")                  #1024,64

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return F.sigmoid(self.conv(dec1))


    def crack_unet_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),

                    (name + "conv2", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),

                    (name + "conv3", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm3", nn.BatchNorm2d(num_features=features)),
                    (name + "relu3", nn.ReLU(inplace=True)),

                    # (name + "conv4", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    # (name + "relu4", nn.ReLU(inplace=True)),


                ]
            )
        )

#Large Crack-Unet : In this, we added one layer to each encoder and decoder block of Crack-Unet.
class Large_Crack_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(Large_Crack_UNet, self).__init__()



        features = init_features
        self.encoder1 = Large_Crack_UNet.crack_unet_block(in_channels, features, name="enc1")          #1024,64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = Large_Crack_UNet.crack_unet_block(features, features * 2, name="enc2")         #512,128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = Large_Crack_UNet.crack_unet_block(features * 2, features * 4, name="enc3")    #256,256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = Large_Crack_UNet.crack_unet_block(features * 4, features * 8, name="enc4")    #128,512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Large_Crack_UNet.crack_unet_block(features * 8, features * 16, name="bottleneck") #64,1024

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)  #128, 512
        self.decoder4 = Large_Crack_UNet.crack_unet_block((features * 8) * 2, features * 8, name="dec4")            #128,512

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2 )   #256, 256
        self.decoder3 = Large_Crack_UNet.crack_unet_block((features * 4) * 2, features * 4, name="dec3")            #256, 256

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2 )  #512,128
        self.decoder2 = Large_Crack_UNet.crack_unet_block((features * 2) * 2, features * 2, name="dec2")           #512,128

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)    #1024,64
        self.decoder1 = Large_Crack_UNet.crack_unet_block(features * 2, features, name="dec1")                  #1024,64

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return F.sigmoid(self.conv(dec1))


    def crack_unet_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),

                    (name + "conv2", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),

                    (name + "conv3", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm3", nn.BatchNorm2d(num_features=features)),
                    (name + "relu3", nn.ReLU(inplace=True)),

                    (name + "conv4", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm4", nn.BatchNorm2d(num_features=features)),
                    (name + "relu4", nn.ReLU(inplace=True)),

                    # (name + "conv4", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    # (name + "relu4", nn.ReLU(inplace=True)),


                ]
            )
        )

#Small Crack-Unet : In this, we remove one layer from each encoder and decoder block of Crack-Unet.
class Small_Crack_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(Small_Crack_UNet, self).__init__()



        features = init_features
        self.encoder1 = Small_Crack_UNet.crack_unet_block(in_channels, features, name="enc1")          #1024,64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = Small_Crack_UNet.crack_unet_block(features, features * 2, name="enc2")         #512,128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = Small_Crack_UNet.crack_unet_block(features * 2, features * 4, name="enc3")    #256,256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = Small_Crack_UNet.crack_unet_block(features * 4, features * 8, name="enc4")    #128,512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Small_Crack_UNet.crack_unet_block(features * 8, features * 16, name="bottleneck") #64,1024

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)  #128, 512
        self.decoder4 = Small_Crack_UNet.crack_unet_block((features * 8) * 2, features * 8, name="dec4")            #128,512

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2 )   #256, 256
        self.decoder3 = Small_Crack_UNet.crack_unet_block((features * 4) * 2, features * 4, name="dec3")            #256, 256

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2 )  #512,128
        self.decoder2 = Small_Crack_UNet.crack_unet_block((features * 2) * 2, features * 2, name="dec2")           #512,128

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)    #1024,64
        self.decoder1 = Small_Crack_UNet.crack_unet_block(features * 2, features, name="dec1")                  #1024,64

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return F.sigmoid(self.conv(dec1))


    def crack_unet_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),

                    (name + "conv2", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),

                    # (name + "conv3", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    # (name + "norm3", nn.BatchNorm2d(num_features=features)),
                    # (name + "relu3", nn.ReLU(inplace=True)),

                    # (name + "conv4", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    # (name + "relu4", nn.ReLU(inplace=True)),


                ]
            )
        )
