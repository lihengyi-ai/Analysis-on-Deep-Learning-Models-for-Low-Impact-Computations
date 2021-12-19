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

#mkl.set_num_threads(4)
np.set_printoptions(threshold = sys.maxsize)#sys.maxsize
#torch.set_printoptions(threshold = 2000,linewidth = 2000)

# Disable parallelism
torch.set_num_threads(1)
# Load pre-trained model
neuralnet = models.vgg19_bn(pretrained=True)
neuralnet = models.shufflenet_v2_x1_0(pretrained=True)
neuralnet = models.mnasnet1_0(pretrained=True)
neuralnet = models.googlenet(pretrained=True)
neuralnet = models.inception_v3(pretrained=True)
#neuralnet = models.densenet121(pretrained=True)
neuralnet = models.resnet18(pretrained=True)
neuralnet = models.alexnet(pretrained=True)
neuralnet = models.mobilenet_v2(pretrained=True)
# Use CPU for the inference
neuralnet.cpu()
# Set the model in the evaluation mode
neuralnet.eval()

conv_n = 16
layer_index = range(1,conv_n+1)

MODEL_CONVLAYER_DATA = numpy.zeros((100,len(layer_index)),dtype = numpy.float)

def get_activation_0(number):
    def hook(model, input, output):
        #print('------------ Input -------------')        
        input_data = input[0]
        #print(type(input_data),type(input_test),type(input))
        input_numpy = input_data.detach().numpy()
        #print(input_data.shape)
        np.save("./DATA/"+str(number)+'.npy',input_numpy)
        # input_txt = str(input_numpy)
        # with open("Layer"+layer+"_input.txt","w") as f:
        #   f.write(input_txt)
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
# alexnet
# =============================================================================

# neuralnet.features[0].register_forward_hook(get_activation_0(1))
# neuralnet.features[3].register_forward_hook(get_activation_0(2))
# neuralnet.features[6].register_forward_hook(get_activation_0(3))
# neuralnet.features[8].register_forward_hook(get_activation_0(4))
# neuralnet.features[10].register_forward_hook(get_activation_0(5))

# =============================================================================
# VGG19bn
# =============================================================================

neuralnet.features[0].register_forward_hook(get_activation_0(1))
neuralnet.features[3].register_forward_hook(get_activation_0(2))
neuralnet.features[7].register_forward_hook(get_activation_0(3))
neuralnet.features[10].register_forward_hook(get_activation_0(4))
neuralnet.features[14].register_forward_hook(get_activation_0(5))
neuralnet.features[17].register_forward_hook(get_activation_0(6))
neuralnet.features[20].register_forward_hook(get_activation_0(7))
neuralnet.features[23].register_forward_hook(get_activation_0(8))
neuralnet.features[27].register_forward_hook(get_activation_0(9))
neuralnet.features[30].register_forward_hook(get_activation_0(10))
neuralnet.features[33].register_forward_hook(get_activation_0(11))
neuralnet.features[36].register_forward_hook(get_activation_0(12))
neuralnet.features[40].register_forward_hook(get_activation_0(13))
neuralnet.features[43].register_forward_hook(get_activation_0(14))
neuralnet.features[46].register_forward_hook(get_activation_0(15))
neuralnet.features[49].register_forward_hook(get_activation_0(16))
# =============================================================================
# VGG13bn
# =============================================================================
# neuralnet.features[0].register_forward_hook(get_activation_0(1))
# neuralnet.features[3].register_forward_hook(get_activation_0(2))
# neuralnet.features[7].register_forward_hook(get_activation_0(3))
# neuralnet.features[10].register_forward_hook(get_activation_0(4))
# neuralnet.features[14].register_forward_hook(get_activation_0(5))
# neuralnet.features[17].register_forward_hook(get_activation_0(6))
# neuralnet.features[21].register_forward_hook(get_activation_0(7))
# neuralnet.features[24].register_forward_hook(get_activation_0(8))
# neuralnet.features[28].register_forward_hook(get_activation_0(9))
# neuralnet.features[31].register_forward_hook(get_activation_0(10))

# # =============================================================================
# # VGG13
# # =============================================================================
# neuralnet.features[0].register_forward_hook(get_activation_0(1))
# neuralnet.features[3].register_forward_hook(get_activation_0(2))
# neuralnet.features[7].register_forward_hook(get_activation_0(3))
# neuralnet.features[10].register_forward_hook(get_activation_0(4))
# neuralnet.features[14].register_forward_hook(get_activation_0(5))
# neuralnet.features[17].register_forward_hook(get_activation_0(6))
# neuralnet.features[21].register_forward_hook(get_activation_0(7))
# neuralnet.features[24].register_forward_hook(get_activation_0(8))
# neuralnet.features[28].register_forward_hook(get_activation_0(9))
# neuralnet.features[31].register_forward_hook(get_activation_0(10))

# =============================================================================
# VGG16
# =============================================================================
# neuralnet.features[0].register_forward_hook(get_activation_0(1))
# neuralnet.features[2].register_forward_hook(get_activation_0(2))
# neuralnet.features[5].register_forward_hook(get_activation_0(3))
# neuralnet.features[7].register_forward_hook(get_activation_0(4))
# neuralnet.features[10].register_forward_hook(get_activation_0(5))
# neuralnet.features[12].register_forward_hook(get_activation_0(6))
# neuralnet.features[14].register_forward_hook(get_activation_0(7))
# neuralnet.features[16].register_forward_hook(get_activation_0(8))
# neuralnet.features[19].register_forward_hook(get_activation_0(9))
# neuralnet.features[21].register_forward_hook(get_activation_0(10))
# neuralnet.features[24].register_forward_hook(get_activation_0(11))
# neuralnet.features[26].register_forward_hook(get_activation_0(12))
# neuralnet.features[28].register_forward_hook(get_activation_0(13))


# =============================================================================
# google
# =============================================================================
# neuralnet.conv1.conv.register_forward_hook(get_activation_0(1))
# neuralnet.conv2.conv.register_forward_hook(get_activation_0(2))
# neuralnet.conv3.conv.register_forward_hook(get_activation_0(3))

# neuralnet.inception3a.branch1.conv.register_forward_hook(get_activation_0(4))
# neuralnet.inception3a.branch2[0].conv.register_forward_hook(get_activation_0(5))
# neuralnet.inception3a.branch2[1].conv.register_forward_hook(get_activation_0(6))
# neuralnet.inception3a.branch3[0].conv.register_forward_hook(get_activation_0(7))
# neuralnet.inception3a.branch3[1].conv.register_forward_hook(get_activation_0(8))
# neuralnet.inception3a.branch4[1].conv.register_forward_hook(get_activation_0(9))

# neuralnet.inception3b.branch1.conv.register_forward_hook(get_activation_0(10))
# neuralnet.inception3b.branch2[0].conv.register_forward_hook(get_activation_0(11))
# neuralnet.inception3b.branch2[1].conv.register_forward_hook(get_activation_0(12))
# neuralnet.inception3b.branch3[0].conv.register_forward_hook(get_activation_0(13))
# neuralnet.inception3b.branch3[1].conv.register_forward_hook(get_activation_0(14))
# neuralnet.inception3b.branch4[1].conv.register_forward_hook(get_activation_0(15))

# neuralnet.inception4a.branch1.conv.register_forward_hook(get_activation_0(16))
# neuralnet.inception4a.branch2[0].conv.register_forward_hook(get_activation_0(17))
# neuralnet.inception4a.branch2[1].conv.register_forward_hook(get_activation_0(18))
# neuralnet.inception4a.branch3[0].conv.register_forward_hook(get_activation_0(19))
# neuralnet.inception4a.branch3[1].conv.register_forward_hook(get_activation_0(20))
# neuralnet.inception4a.branch4[1].conv.register_forward_hook(get_activation_0(21))

# neuralnet.inception4b.branch1.conv.register_forward_hook(get_activation_0(22))
# neuralnet.inception4b.branch2[0].conv.register_forward_hook(get_activation_0(23))
# neuralnet.inception4b.branch2[1].conv.register_forward_hook(get_activation_0(24))
# neuralnet.inception4b.branch3[0].conv.register_forward_hook(get_activation_0(25))
# neuralnet.inception4b.branch3[1].conv.register_forward_hook(get_activation_0(26))
# neuralnet.inception4b.branch4[1].conv.register_forward_hook(get_activation_0(27))


# neuralnet.inception4c.branch1.conv.register_forward_hook(get_activation_0(28))
# neuralnet.inception4c.branch2[0].conv.register_forward_hook(get_activation_0(29))
# neuralnet.inception4c.branch2[1].conv.register_forward_hook(get_activation_0(30))
# neuralnet.inception4c.branch3[0].conv.register_forward_hook(get_activation_0(31))
# neuralnet.inception4c.branch3[1].conv.register_forward_hook(get_activation_0(32))
# neuralnet.inception4c.branch4[1].conv.register_forward_hook(get_activation_0(33))

# neuralnet.inception4d.branch1.conv.register_forward_hook(get_activation_0(34))
# neuralnet.inception4d.branch2[0].conv.register_forward_hook(get_activation_0(35))
# neuralnet.inception4d.branch2[1].conv.register_forward_hook(get_activation_0(36))
# neuralnet.inception4d.branch3[0].conv.register_forward_hook(get_activation_0(37))
# neuralnet.inception4d.branch3[1].conv.register_forward_hook(get_activation_0(38))
# neuralnet.inception4d.branch4[1].conv.register_forward_hook(get_activation_0(39))

# neuralnet.inception4e.branch1.conv.register_forward_hook(get_activation_0(40))
# neuralnet.inception4e.branch2[0].conv.register_forward_hook(get_activation_0(41))
# neuralnet.inception4e.branch2[1].conv.register_forward_hook(get_activation_0(42))
# neuralnet.inception4e.branch3[0].conv.register_forward_hook(get_activation_0(43))
# neuralnet.inception4e.branch3[1].conv.register_forward_hook(get_activation_0(44))
# neuralnet.inception4e.branch4[1].conv.register_forward_hook(get_activation_0(45))

# neuralnet.inception5a.branch1.conv.register_forward_hook(get_activation_0(46))
# neuralnet.inception5a.branch2[0].conv.register_forward_hook(get_activation_0(47))
# neuralnet.inception5a.branch2[1].conv.register_forward_hook(get_activation_0(48))
# neuralnet.inception5a.branch3[0].conv.register_forward_hook(get_activation_0(49))
# neuralnet.inception5a.branch3[1].conv.register_forward_hook(get_activation_0(50))
# neuralnet.inception5a.branch4[1].conv.register_forward_hook(get_activation_0(51))

# neuralnet.inception5b.branch1.conv.register_forward_hook(get_activation_0(52))
# neuralnet.inception5b.branch2[0].conv.register_forward_hook(get_activation_0(53))
# neuralnet.inception5b.branch2[1].conv.register_forward_hook(get_activation_0(54))
# neuralnet.inception5b.branch3[0].conv.register_forward_hook(get_activation_0(55))
# neuralnet.inception5b.branch3[1].conv.register_forward_hook(get_activation_0(56))

# neuralnet.inception5b.branch4[1].conv.register_forward_hook(get_activation_0(57))


# =============================================================================
# inception
# =============================================================================

# neuralnet.Conv2d_1a_3x3.conv.register_forward_hook(get_activation_0(1))
# neuralnet.Conv2d_2a_3x3.conv.register_forward_hook(get_activation_0(2))
# neuralnet.Conv2d_2b_3x3.conv.register_forward_hook(get_activation_0(3))
# neuralnet.Conv2d_3b_1x1.conv.register_forward_hook(get_activation_0(4))
# neuralnet.Conv2d_4a_3x3.conv.register_forward_hook(get_activation_0(5))

# neuralnet.Mixed_5b.branch1x1.conv.register_forward_hook(get_activation_0(6))
# neuralnet.Mixed_5b.branch5x5_1.conv.register_forward_hook(get_activation_0(7))
# neuralnet.Mixed_5b.branch5x5_2.conv.register_forward_hook(get_activation_0(8))

# neuralnet.Mixed_5b.branch3x3dbl_1.conv.register_forward_hook(get_activation_0(9))
# neuralnet.Mixed_5b.branch3x3dbl_2.conv.register_forward_hook(get_activation_0(10))
# neuralnet.Mixed_5b.branch3x3dbl_3.conv.register_forward_hook(get_activation_0(11))

# neuralnet.Mixed_5b.branch_pool.conv.register_forward_hook(get_activation_0(12))


# neuralnet.Mixed_5c.branch1x1.conv.register_forward_hook(get_activation_0(13))
# neuralnet.Mixed_5c.branch5x5_1.conv.register_forward_hook(get_activation_0(14))
# neuralnet.Mixed_5c.branch5x5_2.conv.register_forward_hook(get_activation_0(15))

# neuralnet.Mixed_5c.branch3x3dbl_1.conv.register_forward_hook(get_activation_0(16))
# neuralnet.Mixed_5c.branch3x3dbl_2.conv.register_forward_hook(get_activation_0(17))
# neuralnet.Mixed_5c.branch3x3dbl_3.conv.register_forward_hook(get_activation_0(18))

# neuralnet.Mixed_5c.branch_pool.conv.register_forward_hook(get_activation_0(19))


# neuralnet.Mixed_5d.branch1x1.conv.register_forward_hook(get_activation_0(20))
# neuralnet.Mixed_5d.branch5x5_1.conv.register_forward_hook(get_activation_0(21))
# neuralnet.Mixed_5d.branch5x5_2.conv.register_forward_hook(get_activation_0(22))

# neuralnet.Mixed_5d.branch3x3dbl_1.conv.register_forward_hook(get_activation_0(23))
# neuralnet.Mixed_5d.branch3x3dbl_2.conv.register_forward_hook(get_activation_0(24))
# neuralnet.Mixed_5d.branch3x3dbl_3.conv.register_forward_hook(get_activation_0(25))

# neuralnet.Mixed_5d.branch_pool.conv.register_forward_hook(get_activation_0(26))


# neuralnet.Mixed_6a.branch3x3.conv.register_forward_hook(get_activation_0(27))
# neuralnet.Mixed_6a.branch3x3dbl_1.conv.register_forward_hook(get_activation_0(28))
# neuralnet.Mixed_6a.branch3x3dbl_2.conv.register_forward_hook(get_activation_0(29))
# neuralnet.Mixed_6a.branch3x3dbl_3.conv.register_forward_hook(get_activation_0(30))

# neuralnet.Mixed_6b.branch1x1.conv.register_forward_hook(get_activation_0(31))
# neuralnet.Mixed_6b.branch7x7_1.conv.register_forward_hook(get_activation_0(32))
# neuralnet.Mixed_6b.branch7x7_2.conv.register_forward_hook(get_activation_0(33))
# neuralnet.Mixed_6b.branch7x7_3.conv.register_forward_hook(get_activation_0(34))
# neuralnet.Mixed_6b.branch7x7dbl_1.conv.register_forward_hook(get_activation_0(35))
# neuralnet.Mixed_6b.branch7x7dbl_2.conv.register_forward_hook(get_activation_0(36))
# neuralnet.Mixed_6b.branch7x7dbl_3.conv.register_forward_hook(get_activation_0(37))
# neuralnet.Mixed_6b.branch7x7dbl_4.conv.register_forward_hook(get_activation_0(38))
# neuralnet.Mixed_6b.branch7x7dbl_5.conv.register_forward_hook(get_activation_0(39))

# neuralnet.Mixed_6b.branch_pool.conv.register_forward_hook(get_activation_0(40))

# neuralnet.Mixed_6c.branch1x1.conv.register_forward_hook(get_activation_0(41))
# neuralnet.Mixed_6c.branch7x7_1.conv.register_forward_hook(get_activation_0(42))
# neuralnet.Mixed_6c.branch7x7_2.conv.register_forward_hook(get_activation_0(43))
# neuralnet.Mixed_6c.branch7x7_3.conv.register_forward_hook(get_activation_0(44))
# neuralnet.Mixed_6c.branch7x7dbl_1.conv.register_forward_hook(get_activation_0(45))
# neuralnet.Mixed_6c.branch7x7dbl_2.conv.register_forward_hook(get_activation_0(46))
# neuralnet.Mixed_6c.branch7x7dbl_3.conv.register_forward_hook(get_activation_0(47))
# neuralnet.Mixed_6c.branch7x7dbl_4.conv.register_forward_hook(get_activation_0(48))
# neuralnet.Mixed_6c.branch7x7dbl_5.conv.register_forward_hook(get_activation_0(49))

# neuralnet.Mixed_6c.branch_pool.conv.register_forward_hook(get_activation_0(50))


# neuralnet.Mixed_6d.branch1x1.conv.register_forward_hook(get_activation_0(51))
# neuralnet.Mixed_6d.branch7x7_1.conv.register_forward_hook(get_activation_0(52))
# neuralnet.Mixed_6d.branch7x7_2.conv.register_forward_hook(get_activation_0(53))
# neuralnet.Mixed_6d.branch7x7_3.conv.register_forward_hook(get_activation_0(54))
# neuralnet.Mixed_6d.branch7x7dbl_1.conv.register_forward_hook(get_activation_0(55))
# neuralnet.Mixed_6d.branch7x7dbl_2.conv.register_forward_hook(get_activation_0(56))
# neuralnet.Mixed_6d.branch7x7dbl_3.conv.register_forward_hook(get_activation_0(57))
# neuralnet.Mixed_6d.branch7x7dbl_4.conv.register_forward_hook(get_activation_0(58))
# neuralnet.Mixed_6d.branch7x7dbl_5.conv.register_forward_hook(get_activation_0(59))

# neuralnet.Mixed_6d.branch_pool.conv.register_forward_hook(get_activation_0(60))


# neuralnet.Mixed_6e.branch1x1.conv.register_forward_hook(get_activation_0(61))
# neuralnet.Mixed_6e.branch7x7_1.conv.register_forward_hook(get_activation_0(62))
# neuralnet.Mixed_6e.branch7x7_2.conv.register_forward_hook(get_activation_0(63))
# neuralnet.Mixed_6e.branch7x7_3.conv.register_forward_hook(get_activation_0(64))
# neuralnet.Mixed_6e.branch7x7dbl_1.conv.register_forward_hook(get_activation_0(65))
# neuralnet.Mixed_6e.branch7x7dbl_2.conv.register_forward_hook(get_activation_0(66))
# neuralnet.Mixed_6e.branch7x7dbl_3.conv.register_forward_hook(get_activation_0(67))
# neuralnet.Mixed_6e.branch7x7dbl_4.conv.register_forward_hook(get_activation_0(68))
# neuralnet.Mixed_6e.branch7x7dbl_5.conv.register_forward_hook(get_activation_0(69))

# neuralnet.Mixed_6e.branch_pool.conv.register_forward_hook(get_activation_0(70))

# neuralnet.AuxLogits.conv0.conv.register_forward_hook(get_activation_0(71))
# neuralnet.AuxLogits.conv1.conv.register_forward_hook(get_activation_0(72))

# neuralnet.Mixed_7a.branch3x3_1.conv.register_forward_hook(get_activation_0(73))
# neuralnet.Mixed_7a.branch3x3_2.conv.register_forward_hook(get_activation_0(74))
# neuralnet.Mixed_7a.branch7x7x3_1.conv.register_forward_hook(get_activation_0(75))
# neuralnet.Mixed_7a.branch7x7x3_2.conv.register_forward_hook(get_activation_0(76))
# neuralnet.Mixed_7a.branch7x7x3_3.conv.register_forward_hook(get_activation_0(77))
# neuralnet.Mixed_7a.branch7x7x3_4.conv.register_forward_hook(get_activation_0(78))


# neuralnet.Mixed_7b.branch1x1.conv.register_forward_hook(get_activation_0(79))

# neuralnet.Mixed_7b.branch3x3_1.conv.register_forward_hook(get_activation_0(80))
# neuralnet.Mixed_7b.branch3x3_2a.conv.register_forward_hook(get_activation_0(81))
# neuralnet.Mixed_7b.branch3x3_2b.conv.register_forward_hook(get_activation_0(82))

# neuralnet.Mixed_7b.branch3x3dbl_1.conv.register_forward_hook(get_activation_0(83))
# neuralnet.Mixed_7b.branch3x3dbl_2.conv.register_forward_hook(get_activation_0(84))

# neuralnet.Mixed_7b.branch3x3dbl_3a.conv.register_forward_hook(get_activation_0(85))
# neuralnet.Mixed_7b.branch3x3dbl_3b.conv.register_forward_hook(get_activation_0(86))

# neuralnet.Mixed_7b.branch_pool.conv.register_forward_hook(get_activation_0(87))

# neuralnet.Mixed_7c.branch1x1.conv.register_forward_hook(get_activation_0(88))

# neuralnet.Mixed_7c.branch3x3_1.conv.register_forward_hook(get_activation_0(89))
# neuralnet.Mixed_7c.branch3x3_2a.conv.register_forward_hook(get_activation_0(90))
# neuralnet.Mixed_7c.branch3x3_2b.conv.register_forward_hook(get_activation_0(91))

# neuralnet.Mixed_7c.branch3x3dbl_1.conv.register_forward_hook(get_activation_0(92))
# neuralnet.Mixed_7c.branch3x3dbl_2.conv.register_forward_hook(get_activation_0(93))

# neuralnet.Mixed_7c.branch3x3dbl_3a.conv.register_forward_hook(get_activation_0(94))
# neuralnet.Mixed_7c.branch3x3dbl_3b.conv.register_forward_hook(get_activation_0(95))

# neuralnet.Mixed_7c.branch_pool.conv.register_forward_hook(get_activation_0(96))

# =============================================================================
# densnet121
# =============================================================================

# neuralnet.features.conv0.register_forward_hook(get_activation_0(1))
# neuralnet.features.denseblock1.denselayer1.conv1.register_forward_hook(get_activation_0(2))
# neuralnet.features.denseblock1.denselayer1.conv2.register_forward_hook(get_activation_0(3))
# neuralnet.features.denseblock1.denselayer2.conv1.register_forward_hook(get_activation_0(4))
# neuralnet.features.denseblock1.denselayer2.conv2.register_forward_hook(get_activation_0(5))
# neuralnet.features.denseblock1.denselayer3.conv1.register_forward_hook(get_activation_0(6))
# neuralnet.features.denseblock1.denselayer3.conv2.register_forward_hook(get_activation_0(7))
# neuralnet.features.denseblock1.denselayer4.conv1.register_forward_hook(get_activation_0(8))
# neuralnet.features.denseblock1.denselayer4.conv2.register_forward_hook(get_activation_0(9))
# neuralnet.features.denseblock1.denselayer5.conv1.register_forward_hook(get_activation_0(10))
# neuralnet.features.denseblock1.denselayer5.conv2.register_forward_hook(get_activation_0(11))
# neuralnet.features.denseblock1.denselayer6.conv1.register_forward_hook(get_activation_0(12))
# neuralnet.features.denseblock1.denselayer6.conv2.register_forward_hook(get_activation_0(13))
# neuralnet.features.transition1.conv.register_forward_hook(get_activation_0(14))
# neuralnet.features.denseblock2.denselayer1.conv1.register_forward_hook(get_activation_0(15))
# neuralnet.features.denseblock2.denselayer1.conv2.register_forward_hook(get_activation_0(16))
# neuralnet.features.denseblock2.denselayer2.conv1.register_forward_hook(get_activation_0(17))
# neuralnet.features.denseblock2.denselayer2.conv2.register_forward_hook(get_activation_0(18))
# neuralnet.features.denseblock2.denselayer3.conv1.register_forward_hook(get_activation_0(19))
# neuralnet.features.denseblock2.denselayer3.conv2.register_forward_hook(get_activation_0(20))
# neuralnet.features.denseblock2.denselayer4.conv1.register_forward_hook(get_activation_0(21))
# neuralnet.features.denseblock2.denselayer4.conv2.register_forward_hook(get_activation_0(22))
# neuralnet.features.denseblock2.denselayer5.conv1.register_forward_hook(get_activation_0(23))
# neuralnet.features.denseblock2.denselayer5.conv2.register_forward_hook(get_activation_0(24))
# neuralnet.features.denseblock2.denselayer6.conv1.register_forward_hook(get_activation_0(25))
# neuralnet.features.denseblock2.denselayer6.conv2.register_forward_hook(get_activation_0(26))
# neuralnet.features.denseblock2.denselayer7.conv1.register_forward_hook(get_activation_0(27))
# neuralnet.features.denseblock2.denselayer7.conv2.register_forward_hook(get_activation_0(28))
# neuralnet.features.denseblock2.denselayer8.conv1.register_forward_hook(get_activation_0(29))
# neuralnet.features.denseblock2.denselayer8.conv2.register_forward_hook(get_activation_0(30))
# neuralnet.features.denseblock2.denselayer9.conv1.register_forward_hook(get_activation_0(31))
# neuralnet.features.denseblock2.denselayer9.conv2.register_forward_hook(get_activation_0(32))
# neuralnet.features.denseblock2.denselayer10.conv1.register_forward_hook(get_activation_0(33))
# neuralnet.features.denseblock2.denselayer10.conv2.register_forward_hook(get_activation_0(34))
# neuralnet.features.denseblock2.denselayer11.conv1.register_forward_hook(get_activation_0(35))
# neuralnet.features.denseblock2.denselayer11.conv2.register_forward_hook(get_activation_0(36))
# neuralnet.features.denseblock2.denselayer12.conv1.register_forward_hook(get_activation_0(37))
# neuralnet.features.denseblock2.denselayer12.conv2.register_forward_hook(get_activation_0(38))
# neuralnet.features.transition2.conv.register_forward_hook(get_activation_0(39))
# neuralnet.features.denseblock3.denselayer1.conv1.register_forward_hook(get_activation_0(40))
# neuralnet.features.denseblock3.denselayer1.conv2.register_forward_hook(get_activation_0(41))
# neuralnet.features.denseblock3.denselayer2.conv1.register_forward_hook(get_activation_0(42))
# neuralnet.features.denseblock3.denselayer2.conv2.register_forward_hook(get_activation_0(43))
# neuralnet.features.denseblock3.denselayer3.conv1.register_forward_hook(get_activation_0(44))
# neuralnet.features.denseblock3.denselayer3.conv2.register_forward_hook(get_activation_0(45))
# neuralnet.features.denseblock3.denselayer4.conv1.register_forward_hook(get_activation_0(46))
# neuralnet.features.denseblock3.denselayer4.conv2.register_forward_hook(get_activation_0(47))
# neuralnet.features.denseblock3.denselayer5.conv1.register_forward_hook(get_activation_0(48))
# neuralnet.features.denseblock3.denselayer5.conv2.register_forward_hook(get_activation_0(49))
# neuralnet.features.denseblock3.denselayer6.conv1.register_forward_hook(get_activation_0(50))
# neuralnet.features.denseblock3.denselayer6.conv2.register_forward_hook(get_activation_0(51))
# neuralnet.features.denseblock3.denselayer7.conv1.register_forward_hook(get_activation_0(52))
# neuralnet.features.denseblock3.denselayer7.conv2.register_forward_hook(get_activation_0(53))
# neuralnet.features.denseblock3.denselayer8.conv1.register_forward_hook(get_activation_0(54))
# neuralnet.features.denseblock3.denselayer8.conv2.register_forward_hook(get_activation_0(55))
# neuralnet.features.denseblock3.denselayer9.conv1.register_forward_hook(get_activation_0(56))
# neuralnet.features.denseblock3.denselayer9.conv2.register_forward_hook(get_activation_0(57))
# neuralnet.features.denseblock3.denselayer10.conv1.register_forward_hook(get_activation_0(58))
# neuralnet.features.denseblock3.denselayer10.conv2.register_forward_hook(get_activation_0(59))
# neuralnet.features.denseblock3.denselayer11.conv1.register_forward_hook(get_activation_0(60))
# neuralnet.features.denseblock3.denselayer11.conv2.register_forward_hook(get_activation_0(61))
# neuralnet.features.denseblock3.denselayer12.conv1.register_forward_hook(get_activation_0(62))
# neuralnet.features.denseblock3.denselayer12.conv2.register_forward_hook(get_activation_0(63))
# neuralnet.features.denseblock3.denselayer13.conv1.register_forward_hook(get_activation_0(64))
# neuralnet.features.denseblock3.denselayer13.conv2.register_forward_hook(get_activation_0(65))
# neuralnet.features.denseblock3.denselayer14.conv1.register_forward_hook(get_activation_0(66))
# neuralnet.features.denseblock3.denselayer14.conv2.register_forward_hook(get_activation_0(67))
# neuralnet.features.denseblock3.denselayer15.conv1.register_forward_hook(get_activation_0(68))
# neuralnet.features.denseblock3.denselayer15.conv2.register_forward_hook(get_activation_0(69))
# neuralnet.features.denseblock3.denselayer16.conv1.register_forward_hook(get_activation_0(70))
# neuralnet.features.denseblock3.denselayer16.conv2.register_forward_hook(get_activation_0(71))
# neuralnet.features.denseblock3.denselayer17.conv1.register_forward_hook(get_activation_0(72))
# neuralnet.features.denseblock3.denselayer17.conv2.register_forward_hook(get_activation_0(73))
# neuralnet.features.denseblock3.denselayer18.conv1.register_forward_hook(get_activation_0(74))
# neuralnet.features.denseblock3.denselayer18.conv2.register_forward_hook(get_activation_0(75))
# neuralnet.features.denseblock3.denselayer19.conv1.register_forward_hook(get_activation_0(76))
# neuralnet.features.denseblock3.denselayer19.conv2.register_forward_hook(get_activation_0(77))
# neuralnet.features.denseblock3.denselayer20.conv1.register_forward_hook(get_activation_0(78))
# neuralnet.features.denseblock3.denselayer20.conv2.register_forward_hook(get_activation_0(79))
# neuralnet.features.denseblock3.denselayer21.conv1.register_forward_hook(get_activation_0(80))
# neuralnet.features.denseblock3.denselayer21.conv2.register_forward_hook(get_activation_0(81))
# neuralnet.features.denseblock3.denselayer22.conv1.register_forward_hook(get_activation_0(82))
# neuralnet.features.denseblock3.denselayer22.conv2.register_forward_hook(get_activation_0(83))
# neuralnet.features.denseblock3.denselayer23.conv1.register_forward_hook(get_activation_0(84))
# neuralnet.features.denseblock3.denselayer23.conv2.register_forward_hook(get_activation_0(85))
# neuralnet.features.denseblock3.denselayer24.conv1.register_forward_hook(get_activation_0(86))
# neuralnet.features.denseblock3.denselayer24.conv2.register_forward_hook(get_activation_0(87))
# neuralnet.features.transition3.conv.register_forward_hook(get_activation_0(88))
# neuralnet.features.denseblock4.denselayer1.conv1.register_forward_hook(get_activation_0(89))
# neuralnet.features.denseblock4.denselayer1.conv2.register_forward_hook(get_activation_0(90))
# neuralnet.features.denseblock4.denselayer2.conv1.register_forward_hook(get_activation_0(91))
# neuralnet.features.denseblock4.denselayer2.conv2.register_forward_hook(get_activation_0(92))
# neuralnet.features.denseblock4.denselayer3.conv1.register_forward_hook(get_activation_0(93))
# neuralnet.features.denseblock4.denselayer3.conv2.register_forward_hook(get_activation_0(94))
# neuralnet.features.denseblock4.denselayer4.conv1.register_forward_hook(get_activation_0(95))
# neuralnet.features.denseblock4.denselayer4.conv2.register_forward_hook(get_activation_0(96))
# neuralnet.features.denseblock4.denselayer5.conv1.register_forward_hook(get_activation_0(97))
# neuralnet.features.denseblock4.denselayer5.conv2.register_forward_hook(get_activation_0(98))
# neuralnet.features.denseblock4.denselayer6.conv1.register_forward_hook(get_activation_0(99))
# neuralnet.features.denseblock4.denselayer6.conv2.register_forward_hook(get_activation_0(100))
# neuralnet.features.denseblock4.denselayer7.conv1.register_forward_hook(get_activation_0(101))
# neuralnet.features.denseblock4.denselayer7.conv2.register_forward_hook(get_activation_0(102))
# neuralnet.features.denseblock4.denselayer8.conv1.register_forward_hook(get_activation_0(103))
# neuralnet.features.denseblock4.denselayer8.conv2.register_forward_hook(get_activation_0(104))
# neuralnet.features.denseblock4.denselayer9.conv1.register_forward_hook(get_activation_0(105))
# neuralnet.features.denseblock4.denselayer9.conv2.register_forward_hook(get_activation_0(106))
# neuralnet.features.denseblock4.denselayer10.conv1.register_forward_hook(get_activation_0(107))
# neuralnet.features.denseblock4.denselayer10.conv2.register_forward_hook(get_activation_0(108))
# neuralnet.features.denseblock4.denselayer11.conv1.register_forward_hook(get_activation_0(109))
# neuralnet.features.denseblock4.denselayer11.conv2.register_forward_hook(get_activation_0(110))
# neuralnet.features.denseblock4.denselayer12.conv1.register_forward_hook(get_activation_0(111))
# neuralnet.features.denseblock4.denselayer12.conv2.register_forward_hook(get_activation_0(112))
# neuralnet.features.denseblock4.denselayer13.conv1.register_forward_hook(get_activation_0(113))
# neuralnet.features.denseblock4.denselayer13.conv2.register_forward_hook(get_activation_0(114))
# neuralnet.features.denseblock4.denselayer14.conv1.register_forward_hook(get_activation_0(115))
# neuralnet.features.denseblock4.denselayer14.conv2.register_forward_hook(get_activation_0(116))
# neuralnet.features.denseblock4.denselayer15.conv1.register_forward_hook(get_activation_0(117))
# neuralnet.features.denseblock4.denselayer15.conv2.register_forward_hook(get_activation_0(118))
# neuralnet.features.denseblock4.denselayer16.conv1.register_forward_hook(get_activation_0(119))
# neuralnet.features.denseblock4.denselayer16.conv2.register_forward_hook(get_activation_0(120))


# =============================================================================
# densnet161
# =============================================================================

# neuralnet.features.conv0.register_forward_hook(get_activation_0(1))

# neuralnet.features.denseblock1.denselayer1.conv1.register_forward_hook(get_activation_0(2))
# neuralnet.features.denseblock1.denselayer1.conv2.register_forward_hook(get_activation_0(3))
# neuralnet.features.denseblock1.denselayer2.conv1.register_forward_hook(get_activation_0(4))
# neuralnet.features.denseblock1.denselayer2.conv2.register_forward_hook(get_activation_0(5))
# neuralnet.features.denseblock1.denselayer3.conv1.register_forward_hook(get_activation_0(6))
# neuralnet.features.denseblock1.denselayer3.conv2.register_forward_hook(get_activation_0(7))
# neuralnet.features.denseblock1.denselayer4.conv1.register_forward_hook(get_activation_0(8))
# neuralnet.features.denseblock1.denselayer4.conv2.register_forward_hook(get_activation_0(9))
# neuralnet.features.denseblock1.denselayer5.conv1.register_forward_hook(get_activation_0(10))
# neuralnet.features.denseblock1.denselayer5.conv2.register_forward_hook(get_activation_0(11))
# neuralnet.features.denseblock1.denselayer6.conv1.register_forward_hook(get_activation_0(12))
# neuralnet.features.denseblock1.denselayer6.conv2.register_forward_hook(get_activation_0(13))

# neuralnet.features.transition1.conv.register_forward_hook(get_activation_0(14))


# neuralnet.features.denseblock2.denselayer1.conv1.register_forward_hook(get_activation_0(15))
# neuralnet.features.denseblock2.denselayer1.conv2.register_forward_hook(get_activation_0(16))
# neuralnet.features.denseblock2.denselayer2.conv1.register_forward_hook(get_activation_0(17))
# neuralnet.features.denseblock2.denselayer2.conv2.register_forward_hook(get_activation_0(18))
# neuralnet.features.denseblock2.denselayer3.conv1.register_forward_hook(get_activation_0(19))
# neuralnet.features.denseblock2.denselayer3.conv2.register_forward_hook(get_activation_0(20))
# neuralnet.features.denseblock2.denselayer4.conv1.register_forward_hook(get_activation_0(21))
# neuralnet.features.denseblock2.denselayer4.conv2.register_forward_hook(get_activation_0(22))
# neuralnet.features.denseblock2.denselayer5.conv1.register_forward_hook(get_activation_0(23))
# neuralnet.features.denseblock2.denselayer5.conv2.register_forward_hook(get_activation_0(24))
# neuralnet.features.denseblock2.denselayer6.conv1.register_forward_hook(get_activation_0(25))
# neuralnet.features.denseblock2.denselayer6.conv2.register_forward_hook(get_activation_0(26))
# neuralnet.features.denseblock2.denselayer7.conv1.register_forward_hook(get_activation_0(27))
# neuralnet.features.denseblock2.denselayer7.conv2.register_forward_hook(get_activation_0(28))
# neuralnet.features.denseblock2.denselayer8.conv1.register_forward_hook(get_activation_0(29))
# neuralnet.features.denseblock2.denselayer8.conv2.register_forward_hook(get_activation_0(30))
# neuralnet.features.denseblock2.denselayer9.conv1.register_forward_hook(get_activation_0(31))
# neuralnet.features.denseblock2.denselayer9.conv2.register_forward_hook(get_activation_0(32))
# neuralnet.features.denseblock2.denselayer10.conv1.register_forward_hook(get_activation_0(33))
# neuralnet.features.denseblock2.denselayer10.conv2.register_forward_hook(get_activation_0(34))
# neuralnet.features.denseblock2.denselayer11.conv1.register_forward_hook(get_activation_0(35))
# neuralnet.features.denseblock2.denselayer11.conv2.register_forward_hook(get_activation_0(36))
# neuralnet.features.denseblock2.denselayer12.conv1.register_forward_hook(get_activation_0(37))
# neuralnet.features.denseblock2.denselayer12.conv2.register_forward_hook(get_activation_0(38))

# neuralnet.features.transition2.conv.register_forward_hook(get_activation_0(39))

# neuralnet.features.denseblock3.denselayer1.conv1.register_forward_hook(get_activation_0(40))
# neuralnet.features.denseblock3.denselayer1.conv2.register_forward_hook(get_activation_0(41))
# neuralnet.features.denseblock3.denselayer2.conv1.register_forward_hook(get_activation_0(42))
# neuralnet.features.denseblock3.denselayer2.conv2.register_forward_hook(get_activation_0(43))
# neuralnet.features.denseblock3.denselayer3.conv1.register_forward_hook(get_activation_0(44))
# neuralnet.features.denseblock3.denselayer3.conv2.register_forward_hook(get_activation_0(45))
# neuralnet.features.denseblock3.denselayer4.conv1.register_forward_hook(get_activation_0(46))
# neuralnet.features.denseblock3.denselayer4.conv2.register_forward_hook(get_activation_0(47))
# neuralnet.features.denseblock3.denselayer5.conv1.register_forward_hook(get_activation_0(48))
# neuralnet.features.denseblock3.denselayer5.conv2.register_forward_hook(get_activation_0(49))
# neuralnet.features.denseblock3.denselayer6.conv1.register_forward_hook(get_activation_0(50))
# neuralnet.features.denseblock3.denselayer6.conv2.register_forward_hook(get_activation_0(51))
# neuralnet.features.denseblock3.denselayer7.conv1.register_forward_hook(get_activation_0(52))
# neuralnet.features.denseblock3.denselayer7.conv2.register_forward_hook(get_activation_0(53))
# neuralnet.features.denseblock3.denselayer8.conv1.register_forward_hook(get_activation_0(54))
# neuralnet.features.denseblock3.denselayer8.conv2.register_forward_hook(get_activation_0(55))
# neuralnet.features.denseblock3.denselayer9.conv1.register_forward_hook(get_activation_0(56))
# neuralnet.features.denseblock3.denselayer9.conv2.register_forward_hook(get_activation_0(57))
# neuralnet.features.denseblock3.denselayer10.conv1.register_forward_hook(get_activation_0(58))
# neuralnet.features.denseblock3.denselayer10.conv2.register_forward_hook(get_activation_0(59))
# neuralnet.features.denseblock3.denselayer11.conv1.register_forward_hook(get_activation_0(60))
# neuralnet.features.denseblock3.denselayer11.conv2.register_forward_hook(get_activation_0(61))
# neuralnet.features.denseblock3.denselayer12.conv1.register_forward_hook(get_activation_0(62))
# neuralnet.features.denseblock3.denselayer12.conv2.register_forward_hook(get_activation_0(63))
# neuralnet.features.denseblock3.denselayer13.conv1.register_forward_hook(get_activation_0(64))
# neuralnet.features.denseblock3.denselayer13.conv2.register_forward_hook(get_activation_0(65))
# neuralnet.features.denseblock3.denselayer14.conv1.register_forward_hook(get_activation_0(66))
# neuralnet.features.denseblock3.denselayer14.conv2.register_forward_hook(get_activation_0(67))
# neuralnet.features.denseblock3.denselayer15.conv1.register_forward_hook(get_activation_0(68))
# neuralnet.features.denseblock3.denselayer15.conv2.register_forward_hook(get_activation_0(69))
# neuralnet.features.denseblock3.denselayer16.conv1.register_forward_hook(get_activation_0(70))
# neuralnet.features.denseblock3.denselayer16.conv2.register_forward_hook(get_activation_0(71))
# neuralnet.features.denseblock3.denselayer17.conv1.register_forward_hook(get_activation_0(72))
# neuralnet.features.denseblock3.denselayer17.conv2.register_forward_hook(get_activation_0(73))
# neuralnet.features.denseblock3.denselayer18.conv1.register_forward_hook(get_activation_0(74))
# neuralnet.features.denseblock3.denselayer18.conv2.register_forward_hook(get_activation_0(75))
# neuralnet.features.denseblock3.denselayer19.conv1.register_forward_hook(get_activation_0(76))
# neuralnet.features.denseblock3.denselayer19.conv2.register_forward_hook(get_activation_0(77))
# neuralnet.features.denseblock3.denselayer20.conv1.register_forward_hook(get_activation_0(78))
# neuralnet.features.denseblock3.denselayer20.conv2.register_forward_hook(get_activation_0(79))
# neuralnet.features.denseblock3.denselayer21.conv1.register_forward_hook(get_activation_0(80))
# neuralnet.features.denseblock3.denselayer21.conv2.register_forward_hook(get_activation_0(81))
# neuralnet.features.denseblock3.denselayer22.conv1.register_forward_hook(get_activation_0(82))
# neuralnet.features.denseblock3.denselayer22.conv2.register_forward_hook(get_activation_0(83))
# neuralnet.features.denseblock3.denselayer23.conv1.register_forward_hook(get_activation_0(84))
# neuralnet.features.denseblock3.denselayer23.conv2.register_forward_hook(get_activation_0(85))
# neuralnet.features.denseblock3.denselayer24.conv1.register_forward_hook(get_activation_0(86))
# neuralnet.features.denseblock3.denselayer24.conv2.register_forward_hook(get_activation_0(87))

# neuralnet.features.denseblock3.denselayer25.conv1.register_forward_hook(get_activation_0(88))
# neuralnet.features.denseblock3.denselayer25.conv2.register_forward_hook(get_activation_0(89))
# neuralnet.features.denseblock3.denselayer26.conv1.register_forward_hook(get_activation_0(90))
# neuralnet.features.denseblock3.denselayer26.conv2.register_forward_hook(get_activation_0(91))
# neuralnet.features.denseblock3.denselayer27.conv1.register_forward_hook(get_activation_0(92))
# neuralnet.features.denseblock3.denselayer27.conv2.register_forward_hook(get_activation_0(93))
# neuralnet.features.denseblock3.denselayer28.conv1.register_forward_hook(get_activation_0(94))
# neuralnet.features.denseblock3.denselayer28.conv2.register_forward_hook(get_activation_0(95))

# neuralnet.features.denseblock3.denselayer29.conv1.register_forward_hook(get_activation_0(96))
# neuralnet.features.denseblock3.denselayer29.conv2.register_forward_hook(get_activation_0(97))
# neuralnet.features.denseblock3.denselayer30.conv1.register_forward_hook(get_activation_0(98))
# neuralnet.features.denseblock3.denselayer30.conv2.register_forward_hook(get_activation_0(99))
# neuralnet.features.denseblock3.denselayer31.conv1.register_forward_hook(get_activation_0(100))
# neuralnet.features.denseblock3.denselayer31.conv2.register_forward_hook(get_activation_0(101))
# neuralnet.features.denseblock3.denselayer32.conv1.register_forward_hook(get_activation_0(102))
# neuralnet.features.denseblock3.denselayer32.conv2.register_forward_hook(get_activation_0(103))

# neuralnet.features.denseblock3.denselayer33.conv1.register_forward_hook(get_activation_0(104))
# neuralnet.features.denseblock3.denselayer33.conv2.register_forward_hook(get_activation_0(105))
# neuralnet.features.denseblock3.denselayer34.conv1.register_forward_hook(get_activation_0(106))
# neuralnet.features.denseblock3.denselayer34.conv2.register_forward_hook(get_activation_0(107))
# neuralnet.features.denseblock3.denselayer35.conv1.register_forward_hook(get_activation_0(108))
# neuralnet.features.denseblock3.denselayer35.conv2.register_forward_hook(get_activation_0(109))
# neuralnet.features.denseblock3.denselayer36.conv1.register_forward_hook(get_activation_0(110))
# neuralnet.features.denseblock3.denselayer36.conv2.register_forward_hook(get_activation_0(111))

# neuralnet.features.transition3.conv.register_forward_hook(get_activation_0(112))

# neuralnet.features.denseblock4.denselayer1.conv1.register_forward_hook(get_activation_0(113))
# neuralnet.features.denseblock4.denselayer1.conv2.register_forward_hook(get_activation_0(114))
# neuralnet.features.denseblock4.denselayer2.conv1.register_forward_hook(get_activation_0(115))
# neuralnet.features.denseblock4.denselayer2.conv2.register_forward_hook(get_activation_0(116))
# neuralnet.features.denseblock4.denselayer3.conv1.register_forward_hook(get_activation_0(117))
# neuralnet.features.denseblock4.denselayer3.conv2.register_forward_hook(get_activation_0(118))
# neuralnet.features.denseblock4.denselayer4.conv1.register_forward_hook(get_activation_0(119))
# neuralnet.features.denseblock4.denselayer4.conv2.register_forward_hook(get_activation_0(120))
# neuralnet.features.denseblock4.denselayer5.conv1.register_forward_hook(get_activation_0(121))
# neuralnet.features.denseblock4.denselayer5.conv2.register_forward_hook(get_activation_0(122))
# neuralnet.features.denseblock4.denselayer6.conv1.register_forward_hook(get_activation_0(123))
# neuralnet.features.denseblock4.denselayer6.conv2.register_forward_hook(get_activation_0(124))
# neuralnet.features.denseblock4.denselayer7.conv1.register_forward_hook(get_activation_0(125))
# neuralnet.features.denseblock4.denselayer7.conv2.register_forward_hook(get_activation_0(126))
# neuralnet.features.denseblock4.denselayer8.conv1.register_forward_hook(get_activation_0(127))
# neuralnet.features.denseblock4.denselayer8.conv2.register_forward_hook(get_activation_0(128))
# neuralnet.features.denseblock4.denselayer9.conv1.register_forward_hook(get_activation_0(129))
# neuralnet.features.denseblock4.denselayer9.conv2.register_forward_hook(get_activation_0(130))
# neuralnet.features.denseblock4.denselayer10.conv1.register_forward_hook(get_activation_0(131))
# neuralnet.features.denseblock4.denselayer10.conv2.register_forward_hook(get_activation_0(132))
# neuralnet.features.denseblock4.denselayer11.conv1.register_forward_hook(get_activation_0(133))
# neuralnet.features.denseblock4.denselayer11.conv2.register_forward_hook(get_activation_0(134))
# neuralnet.features.denseblock4.denselayer12.conv1.register_forward_hook(get_activation_0(135))
# neuralnet.features.denseblock4.denselayer12.conv2.register_forward_hook(get_activation_0(136))
# neuralnet.features.denseblock4.denselayer13.conv1.register_forward_hook(get_activation_0(137))
# neuralnet.features.denseblock4.denselayer13.conv2.register_forward_hook(get_activation_0(138))
# neuralnet.features.denseblock4.denselayer14.conv1.register_forward_hook(get_activation_0(139))
# neuralnet.features.denseblock4.denselayer14.conv2.register_forward_hook(get_activation_0(140))
# neuralnet.features.denseblock4.denselayer15.conv1.register_forward_hook(get_activation_0(141))
# neuralnet.features.denseblock4.denselayer15.conv2.register_forward_hook(get_activation_0(142))
# neuralnet.features.denseblock4.denselayer16.conv1.register_forward_hook(get_activation_0(143))
# neuralnet.features.denseblock4.denselayer16.conv2.register_forward_hook(get_activation_0(144))

# neuralnet.features.denseblock4.denselayer17.conv1.register_forward_hook(get_activation_0(145))
# neuralnet.features.denseblock4.denselayer17.conv2.register_forward_hook(get_activation_0(146))
# neuralnet.features.denseblock4.denselayer18.conv1.register_forward_hook(get_activation_0(147))
# neuralnet.features.denseblock4.denselayer18.conv2.register_forward_hook(get_activation_0(148))
# neuralnet.features.denseblock4.denselayer19.conv1.register_forward_hook(get_activation_0(149))
# neuralnet.features.denseblock4.denselayer19.conv2.register_forward_hook(get_activation_0(150))
# neuralnet.features.denseblock4.denselayer20.conv1.register_forward_hook(get_activation_0(151))
# neuralnet.features.denseblock4.denselayer20.conv2.register_forward_hook(get_activation_0(152))
# neuralnet.features.denseblock4.denselayer21.conv1.register_forward_hook(get_activation_0(153))
# neuralnet.features.denseblock4.denselayer21.conv2.register_forward_hook(get_activation_0(154))
# neuralnet.features.denseblock4.denselayer22.conv1.register_forward_hook(get_activation_0(155))
# neuralnet.features.denseblock4.denselayer22.conv2.register_forward_hook(get_activation_0(156))
# neuralnet.features.denseblock4.denselayer23.conv1.register_forward_hook(get_activation_0(157))
# neuralnet.features.denseblock4.denselayer23.conv2.register_forward_hook(get_activation_0(158))
# neuralnet.features.denseblock4.denselayer24.conv1.register_forward_hook(get_activation_0(159))
# neuralnet.features.denseblock4.denselayer24.conv2.register_forward_hook(get_activation_0(160))

# =============================================================================
# densnet169
# =============================================================================


# neuralnet.features.conv0.register_forward_hook(get_activation_0(1))

# neuralnet.features.denseblock1.denselayer1.conv1.register_forward_hook(get_activation_0(2))
# neuralnet.features.denseblock1.denselayer1.conv2.register_forward_hook(get_activation_0(3))
# neuralnet.features.denseblock1.denselayer2.conv1.register_forward_hook(get_activation_0(4))
# neuralnet.features.denseblock1.denselayer2.conv2.register_forward_hook(get_activation_0(5))
# neuralnet.features.denseblock1.denselayer3.conv1.register_forward_hook(get_activation_0(6))
# neuralnet.features.denseblock1.denselayer3.conv2.register_forward_hook(get_activation_0(7))
# neuralnet.features.denseblock1.denselayer4.conv1.register_forward_hook(get_activation_0(8))
# neuralnet.features.denseblock1.denselayer4.conv2.register_forward_hook(get_activation_0(9))
# neuralnet.features.denseblock1.denselayer5.conv1.register_forward_hook(get_activation_0(10))
# neuralnet.features.denseblock1.denselayer5.conv2.register_forward_hook(get_activation_0(11))
# neuralnet.features.denseblock1.denselayer6.conv1.register_forward_hook(get_activation_0(12))
# neuralnet.features.denseblock1.denselayer6.conv2.register_forward_hook(get_activation_0(13))

# neuralnet.features.transition1.conv.register_forward_hook(get_activation_0(14))


# neuralnet.features.denseblock2.denselayer1.conv1.register_forward_hook(get_activation_0(15))
# neuralnet.features.denseblock2.denselayer1.conv2.register_forward_hook(get_activation_0(16))
# neuralnet.features.denseblock2.denselayer2.conv1.register_forward_hook(get_activation_0(17))
# neuralnet.features.denseblock2.denselayer2.conv2.register_forward_hook(get_activation_0(18))
# neuralnet.features.denseblock2.denselayer3.conv1.register_forward_hook(get_activation_0(19))
# neuralnet.features.denseblock2.denselayer3.conv2.register_forward_hook(get_activation_0(20))
# neuralnet.features.denseblock2.denselayer4.conv1.register_forward_hook(get_activation_0(21))
# neuralnet.features.denseblock2.denselayer4.conv2.register_forward_hook(get_activation_0(22))
# neuralnet.features.denseblock2.denselayer5.conv1.register_forward_hook(get_activation_0(23))
# neuralnet.features.denseblock2.denselayer5.conv2.register_forward_hook(get_activation_0(24))
# neuralnet.features.denseblock2.denselayer6.conv1.register_forward_hook(get_activation_0(25))
# neuralnet.features.denseblock2.denselayer6.conv2.register_forward_hook(get_activation_0(26))
# neuralnet.features.denseblock2.denselayer7.conv1.register_forward_hook(get_activation_0(27))
# neuralnet.features.denseblock2.denselayer7.conv2.register_forward_hook(get_activation_0(28))
# neuralnet.features.denseblock2.denselayer8.conv1.register_forward_hook(get_activation_0(29))
# neuralnet.features.denseblock2.denselayer8.conv2.register_forward_hook(get_activation_0(30))
# neuralnet.features.denseblock2.denselayer9.conv1.register_forward_hook(get_activation_0(31))
# neuralnet.features.denseblock2.denselayer9.conv2.register_forward_hook(get_activation_0(32))
# neuralnet.features.denseblock2.denselayer10.conv1.register_forward_hook(get_activation_0(33))
# neuralnet.features.denseblock2.denselayer10.conv2.register_forward_hook(get_activation_0(34))
# neuralnet.features.denseblock2.denselayer11.conv1.register_forward_hook(get_activation_0(35))
# neuralnet.features.denseblock2.denselayer11.conv2.register_forward_hook(get_activation_0(36))
# neuralnet.features.denseblock2.denselayer12.conv1.register_forward_hook(get_activation_0(37))
# neuralnet.features.denseblock2.denselayer12.conv2.register_forward_hook(get_activation_0(38))

# neuralnet.features.transition2.conv.register_forward_hook(get_activation_0(39))

# neuralnet.features.denseblock3.denselayer1.conv1.register_forward_hook(get_activation_0(40))
# neuralnet.features.denseblock3.denselayer1.conv2.register_forward_hook(get_activation_0(41))
# neuralnet.features.denseblock3.denselayer2.conv1.register_forward_hook(get_activation_0(42))
# neuralnet.features.denseblock3.denselayer2.conv2.register_forward_hook(get_activation_0(43))
# neuralnet.features.denseblock3.denselayer3.conv1.register_forward_hook(get_activation_0(44))
# neuralnet.features.denseblock3.denselayer3.conv2.register_forward_hook(get_activation_0(45))
# neuralnet.features.denseblock3.denselayer4.conv1.register_forward_hook(get_activation_0(46))
# neuralnet.features.denseblock3.denselayer4.conv2.register_forward_hook(get_activation_0(47))
# neuralnet.features.denseblock3.denselayer5.conv1.register_forward_hook(get_activation_0(48))
# neuralnet.features.denseblock3.denselayer5.conv2.register_forward_hook(get_activation_0(49))
# neuralnet.features.denseblock3.denselayer6.conv1.register_forward_hook(get_activation_0(50))
# neuralnet.features.denseblock3.denselayer6.conv2.register_forward_hook(get_activation_0(51))
# neuralnet.features.denseblock3.denselayer7.conv1.register_forward_hook(get_activation_0(52))
# neuralnet.features.denseblock3.denselayer7.conv2.register_forward_hook(get_activation_0(53))
# neuralnet.features.denseblock3.denselayer8.conv1.register_forward_hook(get_activation_0(54))
# neuralnet.features.denseblock3.denselayer8.conv2.register_forward_hook(get_activation_0(55))
# neuralnet.features.denseblock3.denselayer9.conv1.register_forward_hook(get_activation_0(56))
# neuralnet.features.denseblock3.denselayer9.conv2.register_forward_hook(get_activation_0(57))
# neuralnet.features.denseblock3.denselayer10.conv1.register_forward_hook(get_activation_0(58))
# neuralnet.features.denseblock3.denselayer10.conv2.register_forward_hook(get_activation_0(59))
# neuralnet.features.denseblock3.denselayer11.conv1.register_forward_hook(get_activation_0(60))
# neuralnet.features.denseblock3.denselayer11.conv2.register_forward_hook(get_activation_0(61))
# neuralnet.features.denseblock3.denselayer12.conv1.register_forward_hook(get_activation_0(62))
# neuralnet.features.denseblock3.denselayer12.conv2.register_forward_hook(get_activation_0(63))
# neuralnet.features.denseblock3.denselayer13.conv1.register_forward_hook(get_activation_0(64))
# neuralnet.features.denseblock3.denselayer13.conv2.register_forward_hook(get_activation_0(65))
# neuralnet.features.denseblock3.denselayer14.conv1.register_forward_hook(get_activation_0(66))
# neuralnet.features.denseblock3.denselayer14.conv2.register_forward_hook(get_activation_0(67))
# neuralnet.features.denseblock3.denselayer15.conv1.register_forward_hook(get_activation_0(68))
# neuralnet.features.denseblock3.denselayer15.conv2.register_forward_hook(get_activation_0(69))
# neuralnet.features.denseblock3.denselayer16.conv1.register_forward_hook(get_activation_0(70))
# neuralnet.features.denseblock3.denselayer16.conv2.register_forward_hook(get_activation_0(71))
# neuralnet.features.denseblock3.denselayer17.conv1.register_forward_hook(get_activation_0(72))
# neuralnet.features.denseblock3.denselayer17.conv2.register_forward_hook(get_activation_0(73))
# neuralnet.features.denseblock3.denselayer18.conv1.register_forward_hook(get_activation_0(74))
# neuralnet.features.denseblock3.denselayer18.conv2.register_forward_hook(get_activation_0(75))
# neuralnet.features.denseblock3.denselayer19.conv1.register_forward_hook(get_activation_0(76))
# neuralnet.features.denseblock3.denselayer19.conv2.register_forward_hook(get_activation_0(77))
# neuralnet.features.denseblock3.denselayer20.conv1.register_forward_hook(get_activation_0(78))
# neuralnet.features.denseblock3.denselayer20.conv2.register_forward_hook(get_activation_0(79))
# neuralnet.features.denseblock3.denselayer21.conv1.register_forward_hook(get_activation_0(80))
# neuralnet.features.denseblock3.denselayer21.conv2.register_forward_hook(get_activation_0(81))
# neuralnet.features.denseblock3.denselayer22.conv1.register_forward_hook(get_activation_0(82))
# neuralnet.features.denseblock3.denselayer22.conv2.register_forward_hook(get_activation_0(83))
# neuralnet.features.denseblock3.denselayer23.conv1.register_forward_hook(get_activation_0(84))
# neuralnet.features.denseblock3.denselayer23.conv2.register_forward_hook(get_activation_0(85))
# neuralnet.features.denseblock3.denselayer24.conv1.register_forward_hook(get_activation_0(86))
# neuralnet.features.denseblock3.denselayer24.conv2.register_forward_hook(get_activation_0(87))

# neuralnet.features.denseblock3.denselayer25.conv1.register_forward_hook(get_activation_0(88))
# neuralnet.features.denseblock3.denselayer25.conv2.register_forward_hook(get_activation_0(89))
# neuralnet.features.denseblock3.denselayer26.conv1.register_forward_hook(get_activation_0(90))
# neuralnet.features.denseblock3.denselayer26.conv2.register_forward_hook(get_activation_0(91))
# neuralnet.features.denseblock3.denselayer27.conv1.register_forward_hook(get_activation_0(92))
# neuralnet.features.denseblock3.denselayer27.conv2.register_forward_hook(get_activation_0(93))
# neuralnet.features.denseblock3.denselayer28.conv1.register_forward_hook(get_activation_0(94))
# neuralnet.features.denseblock3.denselayer28.conv2.register_forward_hook(get_activation_0(95))

# neuralnet.features.denseblock3.denselayer29.conv1.register_forward_hook(get_activation_0(96))
# neuralnet.features.denseblock3.denselayer29.conv2.register_forward_hook(get_activation_0(97))
# neuralnet.features.denseblock3.denselayer30.conv1.register_forward_hook(get_activation_0(98))
# neuralnet.features.denseblock3.denselayer30.conv2.register_forward_hook(get_activation_0(99))
# neuralnet.features.denseblock3.denselayer31.conv1.register_forward_hook(get_activation_0(100))
# neuralnet.features.denseblock3.denselayer31.conv2.register_forward_hook(get_activation_0(101))
# neuralnet.features.denseblock3.denselayer32.conv1.register_forward_hook(get_activation_0(102))
# neuralnet.features.denseblock3.denselayer32.conv2.register_forward_hook(get_activation_0(103))

# neuralnet.features.denseblock3.denselayer33.conv1.register_forward_hook(get_activation_0(104))
# neuralnet.features.denseblock3.denselayer33.conv2.register_forward_hook(get_activation_0(105))
# neuralnet.features.denseblock3.denselayer34.conv1.register_forward_hook(get_activation_0(106))
# neuralnet.features.denseblock3.denselayer34.conv2.register_forward_hook(get_activation_0(107))
# neuralnet.features.denseblock3.denselayer35.conv1.register_forward_hook(get_activation_0(108))
# neuralnet.features.denseblock3.denselayer35.conv2.register_forward_hook(get_activation_0(109))
# neuralnet.features.denseblock3.denselayer36.conv1.register_forward_hook(get_activation_0(110))
# neuralnet.features.denseblock3.denselayer36.conv2.register_forward_hook(get_activation_0(111))

# neuralnet.features.denseblock3.denselayer37.conv1.register_forward_hook(get_activation_0(112))
# neuralnet.features.denseblock3.denselayer37.conv2.register_forward_hook(get_activation_0(113))
# neuralnet.features.denseblock3.denselayer38.conv1.register_forward_hook(get_activation_0(114))
# neuralnet.features.denseblock3.denselayer38.conv2.register_forward_hook(get_activation_0(115))
# neuralnet.features.denseblock3.denselayer39.conv1.register_forward_hook(get_activation_0(116))
# neuralnet.features.denseblock3.denselayer39.conv2.register_forward_hook(get_activation_0(117))
# neuralnet.features.denseblock3.denselayer40.conv1.register_forward_hook(get_activation_0(118))
# neuralnet.features.denseblock3.denselayer40.conv2.register_forward_hook(get_activation_0(119))

# neuralnet.features.denseblock3.denselayer41.conv1.register_forward_hook(get_activation_0(120))
# neuralnet.features.denseblock3.denselayer41.conv2.register_forward_hook(get_activation_0(121))
# neuralnet.features.denseblock3.denselayer42.conv1.register_forward_hook(get_activation_0(122))
# neuralnet.features.denseblock3.denselayer42.conv2.register_forward_hook(get_activation_0(123))
# neuralnet.features.denseblock3.denselayer43.conv1.register_forward_hook(get_activation_0(124))
# neuralnet.features.denseblock3.denselayer43.conv2.register_forward_hook(get_activation_0(125))
# neuralnet.features.denseblock3.denselayer44.conv1.register_forward_hook(get_activation_0(126))
# neuralnet.features.denseblock3.denselayer44.conv2.register_forward_hook(get_activation_0(127))

# neuralnet.features.denseblock3.denselayer45.conv1.register_forward_hook(get_activation_0(128))
# neuralnet.features.denseblock3.denselayer45.conv2.register_forward_hook(get_activation_0(129))
# neuralnet.features.denseblock3.denselayer46.conv1.register_forward_hook(get_activation_0(130))
# neuralnet.features.denseblock3.denselayer46.conv2.register_forward_hook(get_activation_0(131))
# neuralnet.features.denseblock3.denselayer47.conv1.register_forward_hook(get_activation_0(132))
# neuralnet.features.denseblock3.denselayer47.conv2.register_forward_hook(get_activation_0(133))
# neuralnet.features.denseblock3.denselayer48.conv1.register_forward_hook(get_activation_0(134))
# neuralnet.features.denseblock3.denselayer48.conv2.register_forward_hook(get_activation_0(135))

# neuralnet.features.transition3.conv.register_forward_hook(get_activation_0(136))

# neuralnet.features.denseblock4.denselayer1.conv1.register_forward_hook(get_activation_0(137))
# neuralnet.features.denseblock4.denselayer1.conv2.register_forward_hook(get_activation_0(138))
# neuralnet.features.denseblock4.denselayer2.conv1.register_forward_hook(get_activation_0(139))
# neuralnet.features.denseblock4.denselayer2.conv2.register_forward_hook(get_activation_0(140))
# neuralnet.features.denseblock4.denselayer3.conv1.register_forward_hook(get_activation_0(141))
# neuralnet.features.denseblock4.denselayer3.conv2.register_forward_hook(get_activation_0(142))
# neuralnet.features.denseblock4.denselayer4.conv1.register_forward_hook(get_activation_0(143))
# neuralnet.features.denseblock4.denselayer4.conv2.register_forward_hook(get_activation_0(144))
# neuralnet.features.denseblock4.denselayer5.conv1.register_forward_hook(get_activation_0(145))
# neuralnet.features.denseblock4.denselayer5.conv2.register_forward_hook(get_activation_0(146))
# neuralnet.features.denseblock4.denselayer6.conv1.register_forward_hook(get_activation_0(147))
# neuralnet.features.denseblock4.denselayer6.conv2.register_forward_hook(get_activation_0(148))
# neuralnet.features.denseblock4.denselayer7.conv1.register_forward_hook(get_activation_0(149))
# neuralnet.features.denseblock4.denselayer7.conv2.register_forward_hook(get_activation_0(150))
# neuralnet.features.denseblock4.denselayer8.conv1.register_forward_hook(get_activation_0(151))
# neuralnet.features.denseblock4.denselayer8.conv2.register_forward_hook(get_activation_0(152))
# neuralnet.features.denseblock4.denselayer9.conv1.register_forward_hook(get_activation_0(153))
# neuralnet.features.denseblock4.denselayer9.conv2.register_forward_hook(get_activation_0(154))
# neuralnet.features.denseblock4.denselayer10.conv1.register_forward_hook(get_activation_0(155))
# neuralnet.features.denseblock4.denselayer10.conv2.register_forward_hook(get_activation_0(156))
# neuralnet.features.denseblock4.denselayer11.conv1.register_forward_hook(get_activation_0(157))
# neuralnet.features.denseblock4.denselayer11.conv2.register_forward_hook(get_activation_0(158))
# neuralnet.features.denseblock4.denselayer12.conv1.register_forward_hook(get_activation_0(159))
# neuralnet.features.denseblock4.denselayer12.conv2.register_forward_hook(get_activation_0(160))
# neuralnet.features.denseblock4.denselayer13.conv1.register_forward_hook(get_activation_0(161))
# neuralnet.features.denseblock4.denselayer13.conv2.register_forward_hook(get_activation_0(162))
# neuralnet.features.denseblock4.denselayer14.conv1.register_forward_hook(get_activation_0(163))
# neuralnet.features.denseblock4.denselayer14.conv2.register_forward_hook(get_activation_0(164))
# neuralnet.features.denseblock4.denselayer15.conv1.register_forward_hook(get_activation_0(165))
# neuralnet.features.denseblock4.denselayer15.conv2.register_forward_hook(get_activation_0(166))
# neuralnet.features.denseblock4.denselayer16.conv1.register_forward_hook(get_activation_0(167))
# neuralnet.features.denseblock4.denselayer16.conv2.register_forward_hook(get_activation_0(168))

# neuralnet.features.denseblock4.denselayer17.conv1.register_forward_hook(get_activation_0(169))
# neuralnet.features.denseblock4.denselayer17.conv2.register_forward_hook(get_activation_0(170))
# neuralnet.features.denseblock4.denselayer18.conv1.register_forward_hook(get_activation_0(171))
# neuralnet.features.denseblock4.denselayer18.conv2.register_forward_hook(get_activation_0(172))
# neuralnet.features.denseblock4.denselayer19.conv1.register_forward_hook(get_activation_0(173))
# neuralnet.features.denseblock4.denselayer19.conv2.register_forward_hook(get_activation_0(174))
# neuralnet.features.denseblock4.denselayer20.conv1.register_forward_hook(get_activation_0(175))
# neuralnet.features.denseblock4.denselayer20.conv2.register_forward_hook(get_activation_0(176))
# neuralnet.features.denseblock4.denselayer21.conv1.register_forward_hook(get_activation_0(177))
# neuralnet.features.denseblock4.denselayer21.conv2.register_forward_hook(get_activation_0(178))
# neuralnet.features.denseblock4.denselayer22.conv1.register_forward_hook(get_activation_0(179))
# neuralnet.features.denseblock4.denselayer22.conv2.register_forward_hook(get_activation_0(180))
# neuralnet.features.denseblock4.denselayer23.conv1.register_forward_hook(get_activation_0(181))
# neuralnet.features.denseblock4.denselayer23.conv2.register_forward_hook(get_activation_0(182))
# neuralnet.features.denseblock4.denselayer24.conv1.register_forward_hook(get_activation_0(183))
# neuralnet.features.denseblock4.denselayer24.conv2.register_forward_hook(get_activation_0(184))


# neuralnet.features.denseblock4.denselayer25.conv1.register_forward_hook(get_activation_0(185))
# neuralnet.features.denseblock4.denselayer25.conv2.register_forward_hook(get_activation_0(186))
# neuralnet.features.denseblock4.denselayer26.conv1.register_forward_hook(get_activation_0(187))
# neuralnet.features.denseblock4.denselayer26.conv2.register_forward_hook(get_activation_0(188))
# neuralnet.features.denseblock4.denselayer27.conv1.register_forward_hook(get_activation_0(189))
# neuralnet.features.denseblock4.denselayer27.conv2.register_forward_hook(get_activation_0(190))
# neuralnet.features.denseblock4.denselayer28.conv1.register_forward_hook(get_activation_0(191))
# neuralnet.features.denseblock4.denselayer28.conv2.register_forward_hook(get_activation_0(192))

# neuralnet.features.denseblock4.denselayer29.conv1.register_forward_hook(get_activation_0(193))
# neuralnet.features.denseblock4.denselayer29.conv2.register_forward_hook(get_activation_0(194))
# neuralnet.features.denseblock4.denselayer30.conv1.register_forward_hook(get_activation_0(195))
# neuralnet.features.denseblock4.denselayer30.conv2.register_forward_hook(get_activation_0(196))
# neuralnet.features.denseblock4.denselayer31.conv1.register_forward_hook(get_activation_0(197))
# neuralnet.features.denseblock4.denselayer31.conv2.register_forward_hook(get_activation_0(198))
# neuralnet.features.denseblock4.denselayer32.conv1.register_forward_hook(get_activation_0(199))
# neuralnet.features.denseblock4.denselayer32.conv2.register_forward_hook(get_activation_0(200))


# =============================================================================
# mobilenet
# =============================================================================

# neuralnet.features[0][0].register_forward_hook(get_activation_0(1))
# neuralnet.features[1].conv[0][0].register_forward_hook(get_activation_0(2))
# neuralnet.features[1].conv[1].register_forward_hook(get_activation_0(3))

# neuralnet.features[2].conv[0][0].register_forward_hook(get_activation_0(4))
# neuralnet.features[2].conv[1][0].register_forward_hook(get_activation_0(5))
# neuralnet.features[2].conv[2].register_forward_hook(get_activation_0(6))

# neuralnet.features[3].conv[0][0].register_forward_hook(get_activation_0(7))
# neuralnet.features[3].conv[1][0].register_forward_hook(get_activation_0(8))
# neuralnet.features[3].conv[2].register_forward_hook(get_activation_0(9))

# neuralnet.features[4].conv[0][0].register_forward_hook(get_activation_0(10))
# neuralnet.features[4].conv[1][0].register_forward_hook(get_activation_0(11))
# neuralnet.features[4].conv[2].register_forward_hook(get_activation_0(12))

# neuralnet.features[5].conv[0][0].register_forward_hook(get_activation_0(13))
# neuralnet.features[5].conv[1][0].register_forward_hook(get_activation_0(14))
# neuralnet.features[5].conv[2].register_forward_hook(get_activation_0(15))

# neuralnet.features[6].conv[0][0].register_forward_hook(get_activation_0(16))
# neuralnet.features[6].conv[1][0].register_forward_hook(get_activation_0(17))
# neuralnet.features[6].conv[2].register_forward_hook(get_activation_0(18))

# neuralnet.features[7].conv[0][0].register_forward_hook(get_activation_0(19))
# neuralnet.features[7].conv[1][0].register_forward_hook(get_activation_0(20))
# neuralnet.features[7].conv[2].register_forward_hook(get_activation_0(21))

# neuralnet.features[8].conv[0][0].register_forward_hook(get_activation_0(22))
# neuralnet.features[8].conv[1][0].register_forward_hook(get_activation_0(23))
# neuralnet.features[8].conv[2].register_forward_hook(get_activation_0(24))

# neuralnet.features[9].conv[0][0].register_forward_hook(get_activation_0(25))
# neuralnet.features[9].conv[1][0].register_forward_hook(get_activation_0(26))
# neuralnet.features[9].conv[2].register_forward_hook(get_activation_0(27))

# neuralnet.features[10].conv[0][0].register_forward_hook(get_activation_0(28))
# neuralnet.features[10].conv[1][0].register_forward_hook(get_activation_0(29))
# neuralnet.features[10].conv[2].register_forward_hook(get_activation_0(30))

# neuralnet.features[11].conv[0][0].register_forward_hook(get_activation_0(31))
# neuralnet.features[11].conv[1][0].register_forward_hook(get_activation_0(32))
# neuralnet.features[11].conv[2].register_forward_hook(get_activation_0(33))

# neuralnet.features[12].conv[0][0].register_forward_hook(get_activation_0(34))
# neuralnet.features[12].conv[1][0].register_forward_hook(get_activation_0(35))
# neuralnet.features[12].conv[2].register_forward_hook(get_activation_0(36))

# neuralnet.features[13].conv[0][0].register_forward_hook(get_activation_0(37))
# neuralnet.features[13].conv[1][0].register_forward_hook(get_activation_0(38))
# neuralnet.features[13].conv[2].register_forward_hook(get_activation_0(39))

# neuralnet.features[14].conv[0][0].register_forward_hook(get_activation_0(40))
# neuralnet.features[14].conv[1][0].register_forward_hook(get_activation_0(41))
# neuralnet.features[14].conv[2].register_forward_hook(get_activation_0(42))

# neuralnet.features[15].conv[0][0].register_forward_hook(get_activation_0(43))
# neuralnet.features[15].conv[1][0].register_forward_hook(get_activation_0(44))
# neuralnet.features[15].conv[2].register_forward_hook(get_activation_0(45))

# neuralnet.features[16].conv[0][0].register_forward_hook(get_activation_0(46))
# neuralnet.features[16].conv[1][0].register_forward_hook(get_activation_0(47))
# neuralnet.features[16].conv[2].register_forward_hook(get_activation_0(48))

# neuralnet.features[17].conv[0][0].register_forward_hook(get_activation_0(49))
# neuralnet.features[17].conv[1][0].register_forward_hook(get_activation_0(50))
# neuralnet.features[17].conv[2].register_forward_hook(get_activation_0(51))

# neuralnet.features[18][0].register_forward_hook(get_activation_0(52))



# =============================================================================
# shuff
# =============================================================================

# neuralnet.conv1[0].register_forward_hook(get_activation_0(1))

# neuralnet.stage2[0].branch1[0].register_forward_hook(get_activation_0(2))
# neuralnet.stage2[0].branch1[2].register_forward_hook(get_activation_0(3))
# neuralnet.stage2[0].branch2[0].register_forward_hook(get_activation_0(4))
# neuralnet.stage2[0].branch2[3].register_forward_hook(get_activation_0(5))
# neuralnet.stage2[0].branch2[5].register_forward_hook(get_activation_0(6))

# neuralnet.stage2[1].branch2[0].register_forward_hook(get_activation_0(7))
# neuralnet.stage2[1].branch2[3].register_forward_hook(get_activation_0(8))
# neuralnet.stage2[1].branch2[5].register_forward_hook(get_activation_0(9))

# neuralnet.stage2[2].branch2[0].register_forward_hook(get_activation_0(10))
# neuralnet.stage2[2].branch2[3].register_forward_hook(get_activation_0(11))
# neuralnet.stage2[2].branch2[5].register_forward_hook(get_activation_0(12))

# neuralnet.stage2[3].branch2[0].register_forward_hook(get_activation_0(13))
# neuralnet.stage2[3].branch2[3].register_forward_hook(get_activation_0(14))
# neuralnet.stage2[3].branch2[5].register_forward_hook(get_activation_0(15))

# neuralnet.stage3[0].branch1[0].register_forward_hook(get_activation_0(16))
# neuralnet.stage3[0].branch1[2].register_forward_hook(get_activation_0(17))
# neuralnet.stage3[0].branch2[0].register_forward_hook(get_activation_0(18))
# neuralnet.stage3[0].branch2[3].register_forward_hook(get_activation_0(19))
# neuralnet.stage3[0].branch2[5].register_forward_hook(get_activation_0(20))

# neuralnet.stage3[1].branch2[0].register_forward_hook(get_activation_0(21))
# neuralnet.stage3[1].branch2[3].register_forward_hook(get_activation_0(22))
# neuralnet.stage3[1].branch2[5].register_forward_hook(get_activation_0(23))

# neuralnet.stage3[2].branch2[0].register_forward_hook(get_activation_0(24))
# neuralnet.stage3[2].branch2[3].register_forward_hook(get_activation_0(25))
# neuralnet.stage3[2].branch2[5].register_forward_hook(get_activation_0(26))

# neuralnet.stage3[3].branch2[0].register_forward_hook(get_activation_0(27))
# neuralnet.stage3[3].branch2[3].register_forward_hook(get_activation_0(28))
# neuralnet.stage3[3].branch2[5].register_forward_hook(get_activation_0(29))

# neuralnet.stage3[4].branch2[0].register_forward_hook(get_activation_0(30))
# neuralnet.stage3[4].branch2[3].register_forward_hook(get_activation_0(31))
# neuralnet.stage3[4].branch2[5].register_forward_hook(get_activation_0(32))

# neuralnet.stage3[5].branch2[0].register_forward_hook(get_activation_0(33))
# neuralnet.stage3[5].branch2[3].register_forward_hook(get_activation_0(34))
# neuralnet.stage3[5].branch2[5].register_forward_hook(get_activation_0(35))

# neuralnet.stage3[6].branch2[0].register_forward_hook(get_activation_0(36))
# neuralnet.stage3[6].branch2[3].register_forward_hook(get_activation_0(37))
# neuralnet.stage3[6].branch2[5].register_forward_hook(get_activation_0(38))

# neuralnet.stage3[7].branch2[0].register_forward_hook(get_activation_0(39))
# neuralnet.stage3[7].branch2[3].register_forward_hook(get_activation_0(40))
# neuralnet.stage3[7].branch2[5].register_forward_hook(get_activation_0(41))

# neuralnet.stage4[0].branch1[0].register_forward_hook(get_activation_0(42))
# neuralnet.stage4[0].branch1[2].register_forward_hook(get_activation_0(43))
# neuralnet.stage4[0].branch2[0].register_forward_hook(get_activation_0(44))
# neuralnet.stage4[0].branch2[3].register_forward_hook(get_activation_0(45))
# neuralnet.stage4[0].branch2[5].register_forward_hook(get_activation_0(46))

# neuralnet.stage4[1].branch2[0].register_forward_hook(get_activation_0(47))
# neuralnet.stage4[1].branch2[3].register_forward_hook(get_activation_0(48))
# neuralnet.stage4[1].branch2[5].register_forward_hook(get_activation_0(49))

# neuralnet.stage4[2].branch2[0].register_forward_hook(get_activation_0(50))
# neuralnet.stage4[2].branch2[3].register_forward_hook(get_activation_0(51))
# neuralnet.stage4[2].branch2[5].register_forward_hook(get_activation_0(52))

# neuralnet.stage4[3].branch2[0].register_forward_hook(get_activation_0(53))
# neuralnet.stage4[3].branch2[3].register_forward_hook(get_activation_0(54))
# neuralnet.stage4[3].branch2[5].register_forward_hook(get_activation_0(55))

# neuralnet.conv5[0].register_forward_hook(get_activation_0(56))

# =============================================================================
# mnasnet
# =============================================================================

# neuralnet.layers[0].register_forward_hook(get_activation_0(1))
# neuralnet.layers[3].register_forward_hook(get_activation_0(2))
# neuralnet.layers[6].register_forward_hook(get_activation_0(3))

# neuralnet.layers[8][0].layers[0].register_forward_hook(get_activation_0(4))
# neuralnet.layers[8][0].layers[3].register_forward_hook(get_activation_0(5))
# neuralnet.layers[8][0].layers[6].register_forward_hook(get_activation_0(6))

# neuralnet.layers[8][1].layers[0].register_forward_hook(get_activation_0(7))
# neuralnet.layers[8][1].layers[3].register_forward_hook(get_activation_0(8))
# neuralnet.layers[8][1].layers[6].register_forward_hook(get_activation_0(9))

# neuralnet.layers[8][2].layers[0].register_forward_hook(get_activation_0(10))
# neuralnet.layers[8][2].layers[3].register_forward_hook(get_activation_0(11))
# neuralnet.layers[8][2].layers[6].register_forward_hook(get_activation_0(12))

# neuralnet.layers[9][0].layers[0].register_forward_hook(get_activation_0(13))
# neuralnet.layers[9][0].layers[3].register_forward_hook(get_activation_0(14))
# neuralnet.layers[9][0].layers[6].register_forward_hook(get_activation_0(15))

# neuralnet.layers[9][1].layers[0].register_forward_hook(get_activation_0(16))
# neuralnet.layers[9][1].layers[3].register_forward_hook(get_activation_0(17))
# neuralnet.layers[9][1].layers[6].register_forward_hook(get_activation_0(18))

# neuralnet.layers[9][2].layers[0].register_forward_hook(get_activation_0(19))
# neuralnet.layers[9][2].layers[3].register_forward_hook(get_activation_0(20))
# neuralnet.layers[9][2].layers[6].register_forward_hook(get_activation_0(21))

# neuralnet.layers[10][0].layers[0].register_forward_hook(get_activation_0(22))
# neuralnet.layers[10][0].layers[3].register_forward_hook(get_activation_0(23))
# neuralnet.layers[10][0].layers[6].register_forward_hook(get_activation_0(24))

# neuralnet.layers[10][1].layers[0].register_forward_hook(get_activation_0(25))
# neuralnet.layers[10][1].layers[3].register_forward_hook(get_activation_0(26))
# neuralnet.layers[10][1].layers[6].register_forward_hook(get_activation_0(27))

# neuralnet.layers[10][2].layers[0].register_forward_hook(get_activation_0(28))
# neuralnet.layers[10][2].layers[3].register_forward_hook(get_activation_0(29))
# neuralnet.layers[10][2].layers[6].register_forward_hook(get_activation_0(30))

# neuralnet.layers[11][0].layers[0].register_forward_hook(get_activation_0(31))
# neuralnet.layers[11][0].layers[3].register_forward_hook(get_activation_0(32))
# neuralnet.layers[11][0].layers[6].register_forward_hook(get_activation_0(33))

# neuralnet.layers[11][1].layers[0].register_forward_hook(get_activation_0(34))
# neuralnet.layers[11][1].layers[3].register_forward_hook(get_activation_0(35))
# neuralnet.layers[11][1].layers[6].register_forward_hook(get_activation_0(36))

# neuralnet.layers[12][0].layers[0].register_forward_hook(get_activation_0(37))
# neuralnet.layers[12][0].layers[3].register_forward_hook(get_activation_0(38))
# neuralnet.layers[12][0].layers[6].register_forward_hook(get_activation_0(39))

# neuralnet.layers[12][1].layers[0].register_forward_hook(get_activation_0(40))
# neuralnet.layers[12][1].layers[3].register_forward_hook(get_activation_0(41))
# neuralnet.layers[12][1].layers[6].register_forward_hook(get_activation_0(42))

# neuralnet.layers[12][2].layers[0].register_forward_hook(get_activation_0(43))
# neuralnet.layers[12][2].layers[3].register_forward_hook(get_activation_0(44))
# neuralnet.layers[12][2].layers[6].register_forward_hook(get_activation_0(45))

# neuralnet.layers[12][3].layers[0].register_forward_hook(get_activation_0(46))
# neuralnet.layers[12][3].layers[3].register_forward_hook(get_activation_0(47))
# neuralnet.layers[12][3].layers[6].register_forward_hook(get_activation_0(48))

# neuralnet.layers[13][0].layers[0].register_forward_hook(get_activation_0(49))
# neuralnet.layers[13][0].layers[3].register_forward_hook(get_activation_0(50))
# neuralnet.layers[13][0].layers[6].register_forward_hook(get_activation_0(51))

# neuralnet.layers[14].register_forward_hook(get_activation_0(52))



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

#for img_file in image_names[0:INPUT_N]:
for img_file in image_names[0:100]:
           
    img = Image.open(IMAGE_PATH + '/' + img_file)
    print(img_file)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = neuralnet(batch_t)  
    
    # total_times = 0.0
    # index_image = 0
    # index_layer = 0
    # layers_data = np.load('160.npy')
    # element = len(layers_data[0])*len(layers_data[0][0])*len(layers_data[0][0][0]) 
    # print(index_image)
    # print(element)
    # print(layers_data)
    
    # sys.exit()
    index_layer = 0
    #sys.exit()
    for layer_in in range(1,conv_n+1):
        #print(layer_in)
        layers_data = np.load("./DATA/"+str(layer_in)+'.npy')
        element = len(layers_data[0])*len(layers_data[0][0])*len(layers_data[0][0][0]) 
        NZero_features = 0
        #NZero_features_member = 0
        #N_flag = 0
        #N_zero = 0
        #nzero_array =np.zeros(element,dtype = float,order = 'C')
        #data_index = 0
        for i in range(len(layers_data[0])):
            for j in range(len(layers_data[0][i])):
                for k in range(len(layers_data[0][i][j])): 
                    if(layers_data[0][i][j][k] != 0):
                        #nzero_array[data_index] = layers_data[0][i][j][k]
                        #data_index = data_index + 1
                        NZero_features += 1
        #print('NZero_features',NZero_features)     

        percentage = 1-(NZero_features/element)
        MODEL_CONVLAYER_DATA[index_image][index_layer] =  percentage
        index_layer = index_layer + 1 
        
    print(index_image)
    print(MODEL_CONVLAYER_DATA[index_image])
    #print(element)
    index_image = index_image + 1 
    #print(len(MODEL_CONVLAYER_DATA[index_image]))
    #print(MODEL_CONVLAYER_DATA[index_image])
    #for layer_in in range(1,2):
        #os.remove("./DATA/"+str(layer_in)+'.npy')

    
    #sys.exit()

np.save("./VGG19bn_SPARSITY_DATA.npy",MODEL_CONVLAYER_DATA)
print("over")



