#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:36:29 2021

@author: ihpc
"""

from ctypes import CDLL
import time
import os
import sys
from PIL import Image
import torch
import torchvision
import torchprof
from torchvision import models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import torch.nn as nn
import mkl
import numpy
import ctypes

INPUT_N = 100

# =============================================================================
# test the data
# =============================================================================

np.set_printoptions(threshold = sys.maxsize)#sys.maxsize
#torch.set_printoptions(threshold = 2000,linewidth = 2000)

# Disable parallelism
torch.set_num_threads(1)
# Load pre-trained model
neuralnet = models.resnet18(pretrained=True)
#neuralnet = models.resnext101_32x8d(pretrained=True)

neuralnet.cpu()
# Set the model in the evaluation mode
neuralnet.eval()

layer_index = range(1,21)

MODEL_CONVLAYER_DATA = numpy.zeros((100,len(layer_index)),dtype = numpy.float)

def get_activation_0(number):
    def hook(model, input, output):
        #print('------------ Input -------------')
        input_data =input[0]
        #print(type(input_data),type(input_test),type(input))
        input_numpy = input_data.detach().numpy()
        print(input_data.shape)
        np.save(str(number)+'.npy',input_numpy)
        input_txt = str(input_numpy)
        with open(str(number)+"_input.txt","w") as f:
           f.write(input_txt)
        #print('----------- Output --------------')
        #output_numpy = output.detach().numpy()
        #print('type_out',type(output))
        #print(output.shape)
        # Count_zero_feature = 0
        # for i in range(len(output[0])):
        #     Count_Nzero = 0
        #     for j in range(len(output[0][i])):
        #         for k in range(len(output[0][i][j])):
        #             if (output[0][i][j][k] <= -2):
        #                output[0][i][j][k] = 0.5
        '''
            if (Count_Nzero <= 5):
                for m in range(len(output[0][i])):
                    for n in range(len(output[0][i][m])):
                        output[0][i][m][n] = 0
        '''
        #np.save("Layer"+layer+"_output.npy",output_numpy)
        #output_txt = str(output_numpy)
        #print('output_type',type(output))
        #print(type(output_numpy))
        #print(output.shape)
        # with open("Layer"+layer+"_output.txt","w") as f:
        #   f.write(output_txt) 
    return hook

# =============================================================================
# resnet18
# =============================================================================

# neuralnet.conv1.register_forward_hook(get_activation_0(1))

# neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(2))
# neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(3))
# neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(4))
# neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(5))

# neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(6))
# neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(7))
# neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(8))
# neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(9))
# neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(10))

# neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(11))
# neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(12))
# neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(13))
# neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(14))
# neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(15))

#neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(16))
neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(17))
neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(18))
neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(19))
neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(20))

# =============================================================================
# resnet34
# =============================================================================

# neuralnet.conv1.register_forward_hook(get_activation_0(1))

# neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(2))
# neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(3))
# neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(4))
# neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(5))
# neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(6))
# neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(7))


# neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(8))
# neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(9))
# neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(10))
# neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(11))
# neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(12))
# neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(13))
# neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(14))
# neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(15))
# neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(16))

# neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(17))
# neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(18))
# neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(19))
# neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(20))
# neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(21))
# neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(22))
# neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(23))
# neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(24))
# neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(25))
# neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(26))
# neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(27))
# neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(28))
# neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(29))


# neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(30))
# neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(31))
# neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(32))
# neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(33))
# neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(34))
# neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(35))
# neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(36))


# =============================================================================
# resnet50
# =============================================================================

# neuralnet.conv1.register_forward_hook(get_activation_0(1))

# neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(2))
# neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(3))
# neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(4))
# neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
# neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(6))
# neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(7))
# neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(8))
# neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(9))
# neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(10))
# neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(11))


# neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(12))
# neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(13))
# neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(14))
# neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
# neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(16))
# neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(17))
# neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(18))
# neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(19))
# neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(20))
# neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(21))
# neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(22))
# neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(23))
# neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(24))

# neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(25))
# neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(26))
# neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(27))
# neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(28))
# neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(29))
# neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(30))
# neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(31))
# neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(32))
# neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(33))
# neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(34))
# neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(35))
# neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(36))
# neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(37))
# neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(38))
# neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(39))
# neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(40))
# neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(41))
# neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(42))
# neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(43))


# neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(44))
# neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(45))
# neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(46))
# neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(47))
# neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(48))
# neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(49))
# neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(50))
# neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(51))
# neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(52))
# neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(53))
# ======================================================================

# =============================================================================
# resnet101
# =============================================================================

# neuralnet.conv1.register_forward_hook(get_activation_0(1))

# neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(2))
# neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(3))
# neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(4))
# neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
# neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(6))
# neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(7))
# neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(8))
# neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(9))
# neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(10))
# neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(11))


# neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(12))
# neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(13))
# neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(14))
# neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
# neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(16))
# neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(17))
# neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(18))
# neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(19))
# neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(20))
# neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(21))
# neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(22))
# neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(23))
# neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(24))

# neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(25))
# neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(26))
# neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(27))
# neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(28))
# neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(29))
# neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(30))
# neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(31))
# neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(32))
# neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(33))
# neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(34))
# neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(35))
# neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(36))
# neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(37))
# neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(38))
# neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(39))
# neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(40))
# neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(41))
# neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(42))
# neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(43))

# neuralnet.layer3[6].conv1.register_forward_hook(get_activation_0(44))
# neuralnet.layer3[6].conv2.register_forward_hook(get_activation_0(45))
# neuralnet.layer3[6].conv3.register_forward_hook(get_activation_0(46))
# neuralnet.layer3[7].conv1.register_forward_hook(get_activation_0(47))
# neuralnet.layer3[7].conv2.register_forward_hook(get_activation_0(48))
# neuralnet.layer3[7].conv3.register_forward_hook(get_activation_0(49))
# neuralnet.layer3[8].conv1.register_forward_hook(get_activation_0(50))
# neuralnet.layer3[8].conv2.register_forward_hook(get_activation_0(51))
# neuralnet.layer3[8].conv3.register_forward_hook(get_activation_0(52))
# neuralnet.layer3[9].conv1.register_forward_hook(get_activation_0(53))
# neuralnet.layer3[9].conv2.register_forward_hook(get_activation_0(54))
# neuralnet.layer3[9].conv3.register_forward_hook(get_activation_0(55))
# neuralnet.layer3[10].conv1.register_forward_hook(get_activation_0(56))
# neuralnet.layer3[10].conv2.register_forward_hook(get_activation_0(57))
# neuralnet.layer3[10].conv3.register_forward_hook(get_activation_0(58))
# neuralnet.layer3[11].conv1.register_forward_hook(get_activation_0(59))
# neuralnet.layer3[11].conv2.register_forward_hook(get_activation_0(60))
# neuralnet.layer3[11].conv3.register_forward_hook(get_activation_0(61))
# neuralnet.layer3[12].conv1.register_forward_hook(get_activation_0(62))
# neuralnet.layer3[12].conv2.register_forward_hook(get_activation_0(63))
# neuralnet.layer3[12].conv3.register_forward_hook(get_activation_0(64))
# neuralnet.layer3[13].conv1.register_forward_hook(get_activation_0(65))
# neuralnet.layer3[13].conv2.register_forward_hook(get_activation_0(66))
# neuralnet.layer3[13].conv3.register_forward_hook(get_activation_0(67))
# neuralnet.layer3[14].conv1.register_forward_hook(get_activation_0(68))
# neuralnet.layer3[14].conv2.register_forward_hook(get_activation_0(69))
# neuralnet.layer3[14].conv3.register_forward_hook(get_activation_0(70))
# neuralnet.layer3[15].conv1.register_forward_hook(get_activation_0(71))
# neuralnet.layer3[15].conv2.register_forward_hook(get_activation_0(72))
# neuralnet.layer3[15].conv3.register_forward_hook(get_activation_0(73))
# neuralnet.layer3[16].conv1.register_forward_hook(get_activation_0(74))
# neuralnet.layer3[16].conv2.register_forward_hook(get_activation_0(75))
# neuralnet.layer3[16].conv3.register_forward_hook(get_activation_0(76))
# neuralnet.layer3[17].conv1.register_forward_hook(get_activation_0(77))
# neuralnet.layer3[17].conv2.register_forward_hook(get_activation_0(78))
# neuralnet.layer3[17].conv3.register_forward_hook(get_activation_0(79))
# neuralnet.layer3[18].conv1.register_forward_hook(get_activation_0(80))
# neuralnet.layer3[18].conv2.register_forward_hook(get_activation_0(81))
# neuralnet.layer3[18].conv3.register_forward_hook(get_activation_0(82))
# neuralnet.layer3[19].conv1.register_forward_hook(get_activation_0(83))
# neuralnet.layer3[19].conv2.register_forward_hook(get_activation_0(84))
# neuralnet.layer3[19].conv3.register_forward_hook(get_activation_0(85))
# neuralnet.layer3[20].conv1.register_forward_hook(get_activation_0(86))
# neuralnet.layer3[20].conv2.register_forward_hook(get_activation_0(87))
# neuralnet.layer3[20].conv3.register_forward_hook(get_activation_0(88))
# neuralnet.layer3[21].conv1.register_forward_hook(get_activation_0(89))
# neuralnet.layer3[21].conv2.register_forward_hook(get_activation_0(90))
# neuralnet.layer3[21].conv3.register_forward_hook(get_activation_0(91))
# neuralnet.layer3[22].conv1.register_forward_hook(get_activation_0(92))
# neuralnet.layer3[22].conv2.register_forward_hook(get_activation_0(93))
# neuralnet.layer3[22].conv3.register_forward_hook(get_activation_0(94))

# neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(95))
# neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(96))
# neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(97))
# neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(98))
# neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(99))
# neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(100))
# neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(101))
# neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(102))
# neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(103))
# neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(104))


# =============================================================================
# resnet152
# =============================================================================

# neuralnet.conv1.register_forward_hook(get_activation_0(1))
# neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(2))
# neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(3))
# neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(4))
# neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
# neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(6))
# neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(7))
# neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(8))
# neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(9))
# neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(10))
# neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(11))
# neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(12))
# neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(13))
# neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(14))
# neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
# neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(16))
# neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(17))
# neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(18))
# neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(19))
# neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(20))
# neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(21))
# neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(22))
# neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(23))
# neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(24))
# neuralnet.layer2[4].conv1.register_forward_hook(get_activation_0(25))
# neuralnet.layer2[4].conv2.register_forward_hook(get_activation_0(26))
# neuralnet.layer2[4].conv3.register_forward_hook(get_activation_0(27))
# neuralnet.layer2[5].conv1.register_forward_hook(get_activation_0(28))
# neuralnet.layer2[5].conv2.register_forward_hook(get_activation_0(29))
# neuralnet.layer2[5].conv3.register_forward_hook(get_activation_0(30))
# neuralnet.layer2[6].conv1.register_forward_hook(get_activation_0(31))
# neuralnet.layer2[6].conv2.register_forward_hook(get_activation_0(32))
# neuralnet.layer2[6].conv3.register_forward_hook(get_activation_0(33))
# neuralnet.layer2[7].conv1.register_forward_hook(get_activation_0(34))
# neuralnet.layer2[7].conv2.register_forward_hook(get_activation_0(35))
# neuralnet.layer2[7].conv3.register_forward_hook(get_activation_0(36))
# neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(37))
# neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(38))
# neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(39))
# neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(40))
# neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(41))
# neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(42))
# neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(43))
# neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(44))
# neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(45))
# neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(46))
# neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(47))
# neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(48))
# neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(49))
# neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(50))
# neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(51))
# neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(52))
# neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(53))
# neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(54))
# neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(55))
# neuralnet.layer3[6].conv1.register_forward_hook(get_activation_0(56))
# neuralnet.layer3[6].conv2.register_forward_hook(get_activation_0(57))
# neuralnet.layer3[6].conv3.register_forward_hook(get_activation_0(58))
# neuralnet.layer3[7].conv1.register_forward_hook(get_activation_0(59))
# neuralnet.layer3[7].conv2.register_forward_hook(get_activation_0(60))
# neuralnet.layer3[7].conv3.register_forward_hook(get_activation_0(61))
# neuralnet.layer3[8].conv1.register_forward_hook(get_activation_0(62))
# neuralnet.layer3[8].conv2.register_forward_hook(get_activation_0(63))
# neuralnet.layer3[8].conv3.register_forward_hook(get_activation_0(64))
# neuralnet.layer3[9].conv1.register_forward_hook(get_activation_0(65))
# neuralnet.layer3[9].conv2.register_forward_hook(get_activation_0(66))
# neuralnet.layer3[9].conv3.register_forward_hook(get_activation_0(67))
# neuralnet.layer3[10].conv1.register_forward_hook(get_activation_0(68))
# neuralnet.layer3[10].conv2.register_forward_hook(get_activation_0(69))
# neuralnet.layer3[10].conv3.register_forward_hook(get_activation_0(70))
# neuralnet.layer3[11].conv1.register_forward_hook(get_activation_0(71))
# neuralnet.layer3[11].conv2.register_forward_hook(get_activation_0(72))
# neuralnet.layer3[11].conv3.register_forward_hook(get_activation_0(73))
# neuralnet.layer3[12].conv1.register_forward_hook(get_activation_0(74))
# neuralnet.layer3[12].conv2.register_forward_hook(get_activation_0(75))
# neuralnet.layer3[12].conv3.register_forward_hook(get_activation_0(76))
# neuralnet.layer3[13].conv1.register_forward_hook(get_activation_0(77))
# neuralnet.layer3[13].conv2.register_forward_hook(get_activation_0(78))
# neuralnet.layer3[13].conv3.register_forward_hook(get_activation_0(79))
# neuralnet.layer3[14].conv1.register_forward_hook(get_activation_0(80))
# neuralnet.layer3[14].conv2.register_forward_hook(get_activation_0(81))
# neuralnet.layer3[14].conv3.register_forward_hook(get_activation_0(82))
# neuralnet.layer3[15].conv1.register_forward_hook(get_activation_0(83))
# neuralnet.layer3[15].conv2.register_forward_hook(get_activation_0(84))
# neuralnet.layer3[15].conv3.register_forward_hook(get_activation_0(85))
# neuralnet.layer3[16].conv1.register_forward_hook(get_activation_0(86))
# neuralnet.layer3[16].conv2.register_forward_hook(get_activation_0(87))
# neuralnet.layer3[16].conv3.register_forward_hook(get_activation_0(88))
# neuralnet.layer3[17].conv1.register_forward_hook(get_activation_0(89))
# neuralnet.layer3[17].conv2.register_forward_hook(get_activation_0(90))
# neuralnet.layer3[17].conv3.register_forward_hook(get_activation_0(91))
# neuralnet.layer3[18].conv1.register_forward_hook(get_activation_0(92))
# neuralnet.layer3[18].conv2.register_forward_hook(get_activation_0(93))
# neuralnet.layer3[18].conv3.register_forward_hook(get_activation_0(94))
# neuralnet.layer3[19].conv1.register_forward_hook(get_activation_0(95))
# neuralnet.layer3[19].conv2.register_forward_hook(get_activation_0(96))
# neuralnet.layer3[19].conv3.register_forward_hook(get_activation_0(97))
# neuralnet.layer3[20].conv1.register_forward_hook(get_activation_0(98))
# neuralnet.layer3[20].conv2.register_forward_hook(get_activation_0(99))
# neuralnet.layer3[20].conv3.register_forward_hook(get_activation_0(100))
# neuralnet.layer3[21].conv1.register_forward_hook(get_activation_0(101))
# neuralnet.layer3[21].conv2.register_forward_hook(get_activation_0(102))
# neuralnet.layer3[21].conv3.register_forward_hook(get_activation_0(103))
# neuralnet.layer3[22].conv1.register_forward_hook(get_activation_0(104))
# neuralnet.layer3[22].conv2.register_forward_hook(get_activation_0(105))
# neuralnet.layer3[22].conv3.register_forward_hook(get_activation_0(106))
# neuralnet.layer3[23].conv1.register_forward_hook(get_activation_0(107))
# neuralnet.layer3[23].conv2.register_forward_hook(get_activation_0(108))
# neuralnet.layer3[23].conv3.register_forward_hook(get_activation_0(109))
# neuralnet.layer3[24].conv1.register_forward_hook(get_activation_0(110))
# neuralnet.layer3[24].conv2.register_forward_hook(get_activation_0(111))
# neuralnet.layer3[24].conv3.register_forward_hook(get_activation_0(112))
# neuralnet.layer3[25].conv1.register_forward_hook(get_activation_0(113))
# neuralnet.layer3[25].conv2.register_forward_hook(get_activation_0(114))
# neuralnet.layer3[25].conv3.register_forward_hook(get_activation_0(115))
# neuralnet.layer3[26].conv1.register_forward_hook(get_activation_0(116))
# neuralnet.layer3[26].conv2.register_forward_hook(get_activation_0(117))
# neuralnet.layer3[26].conv3.register_forward_hook(get_activation_0(118))
# neuralnet.layer3[27].conv1.register_forward_hook(get_activation_0(119))
# neuralnet.layer3[27].conv2.register_forward_hook(get_activation_0(120))
# neuralnet.layer3[27].conv3.register_forward_hook(get_activation_0(121))
# neuralnet.layer3[28].conv1.register_forward_hook(get_activation_0(122))
# neuralnet.layer3[28].conv2.register_forward_hook(get_activation_0(123))
# neuralnet.layer3[28].conv3.register_forward_hook(get_activation_0(124))
# neuralnet.layer3[29].conv1.register_forward_hook(get_activation_0(125))
# neuralnet.layer3[29].conv2.register_forward_hook(get_activation_0(126))
# neuralnet.layer3[29].conv3.register_forward_hook(get_activation_0(127))
# neuralnet.layer3[30].conv1.register_forward_hook(get_activation_0(128))
# neuralnet.layer3[30].conv2.register_forward_hook(get_activation_0(129))
# neuralnet.layer3[30].conv3.register_forward_hook(get_activation_0(130))
# neuralnet.layer3[31].conv1.register_forward_hook(get_activation_0(131))
# neuralnet.layer3[31].conv2.register_forward_hook(get_activation_0(132))
# neuralnet.layer3[31].conv3.register_forward_hook(get_activation_0(133))
# neuralnet.layer3[32].conv1.register_forward_hook(get_activation_0(134))
# neuralnet.layer3[32].conv2.register_forward_hook(get_activation_0(135))
# neuralnet.layer3[32].conv3.register_forward_hook(get_activation_0(136))
# neuralnet.layer3[33].conv1.register_forward_hook(get_activation_0(137))
# neuralnet.layer3[33].conv2.register_forward_hook(get_activation_0(138))
# neuralnet.layer3[33].conv3.register_forward_hook(get_activation_0(139))
# neuralnet.layer3[34].conv1.register_forward_hook(get_activation_0(140))
# neuralnet.layer3[34].conv2.register_forward_hook(get_activation_0(141))
# neuralnet.layer3[34].conv3.register_forward_hook(get_activation_0(142))
# neuralnet.layer3[35].conv1.register_forward_hook(get_activation_0(143))
# neuralnet.layer3[35].conv2.register_forward_hook(get_activation_0(144))
# neuralnet.layer3[35].conv3.register_forward_hook(get_activation_0(145))
# neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(146))
# neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(147))
# neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(148))
# neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(149))
# neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(150))
# neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(151))
# neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(152))
# neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(153))
# neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(154))
# neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(155))


# =============================================================================
# Read the labels
# =============================================================================
with open('./synsets.txt') as f0:
    labels = [line for line in f0.readlines()]

with open('./ILSVRC2012_validation_truth_label.txt') as f1:
    truthlabels = [line for line in f1.readlines()]
    
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.Pad(2,0),
    transforms.ToTensor(),  
    transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
])
IMAGE_PATH = './data/ILSVRC2012_val'
count = 0
counter = 0
Right_num = 0
total_times = 0.0
index_image = 0
index_layer = 0
image_names = os.listdir(IMAGE_PATH)
image_names.sort()

for img_file in image_names[0:1]:
           
    print(img_file)
    img = Image.open(IMAGE_PATH + '/' + img_file)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = neuralnet(batch_t)  

    index_layer = 0
    #sys.exit()
    #for layer_in in layer_index[15:len(layer_index)]:
    for layer_in in range(17,20):
        layers_data = np.load(str(layer_in)+'.npy')
        #print(layers_data[0][2])
        element = len(layers_data[0])*len(layers_data[0][0])*len(layers_data[0][0][0]) 
        NZero_features = 0
        NZero_features_member = 0
        N_flag = 0
        N_zero = 0
        nzero_array =np.zeros(element,dtype = float,order = 'C')
        data_index = 0
        for i in range(len(layers_data[0])):
            for j in range(len(layers_data[0][i])):
                for k in range(len(layers_data[0][i][j])): 
                    if(layers_data[0][i][j][k] != 0):
                        nzero_array[data_index] = layers_data[0][i][j][k]
                        data_index = data_index + 1
                        NZero_features +=1
        #print('NZero_features',NZero_features)     

        #print(layer,"percent:",1-NZero_features/element)
        percentage = 1-NZero_features/element
        print("percent",percentage)
        MODEL_CONVLAYER_DATA[index_image][index_layer] =  percentage
        #print(MODEL_CONVLAYER_DATA[0])
        
        index_layer = index_layer + 1 
    
    print(index_image)    
    print(MODEL_CONVLAYER_DATA[index_image])
    
    index_image = index_image + 1
    #sys.exit()

np.save("RES50_SPARSITY_DATA0.npy",MODEL_CONVLAYER_DATA)
print("over")



