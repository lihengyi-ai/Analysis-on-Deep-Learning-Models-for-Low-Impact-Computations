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

#alexnet  25   5
#VGG13 MODEL LAYER: 37; CONV LAYER: 10; 
#VGG16 MODEL LAYER: 43; CONV LAYER: 13; 
#VGG19 MODEL LAYER: 49; CONV LAYER: 16; 

#RESNET18 MODEL LAYER: 70; CONV LAYER: 20; 
#RESNET34 MODEL LAYER: 126; CONV LAYER: 36; 
#RESNET50 MODEL LAYER: 176; CONV LAYER: 53; 
#RESNET101 MODEL LAYER: 346; CONV LAYER: 104; 
#RESNET152 MODEL LAYER: 516; CONV LAYER: 155; 

#googlenet 214 57
#inception_v3 331 96

#densenet121 432 120
#densenet161 572 160
#densenet169 600 168
#densenet169 712 200

#mobilenet_v2: 154; CONV LAYER: 52; 
#mnasnet1_0 153 52
#shufflenet_v2_x0_5 246 56

Input_N = 1000

AVERAGE_MODEL_TIME = 0
torch.autograd.profiler.SUM_MODLE_TIME = 0
torch.autograd.profiler.MODEL_INDEX = 0
torch.autograd.profiler.CONVLAYER_INDEX = 0
torch.autograd.profiler.MODEL_INDEX_N = 0
torch.autograd.profiler.INPUT_INDEX = 0
torch.autograd.profiler.index_line = 0

model_time = np.zeros(1000,dtype = float,order = 'C')
#OMP_NUM_THREADS = 1
np.set_printoptions(threshold = sys.maxsize)#sys.maxsize
#torch.set_printoptions(threshold = 2000,linewidth = 2000)

mkl.set_num_threads(1)
# Disable parallelism
torch.set_num_threads(1)

# Load pre-trained model
neuralnet = models.vgg19_bn(pretrained=True)
#neuralnet = models.mobilenet_v2(pretrained=True)

#neuralnet = models.mnasnet0_5(pretrained=True)
#neuralnet = models.inception_v3(pretrained=True)
#neuralnet = models.densenet169(pretrained=True)
#neuralnet = models.googlenet(pretrained=True)
#neuralnet = models.shufflenet_v2_x0_5(pretrained=True)
#neuralnet = models.shufflenet_v2_x1_0(pretrained=True)
# neuralnet = models.alexnet(pretrained=True)
#neuralnet = models.shufflenet_v2_x1_0(pretrained=True)trash:///pytorch.6/torch/autograd/profiler.py

#neuralnet = models.wide_resnet50_2(pretrained=True)
#neuralnet = models.resnet101(pretrained=True)
# print(neuralnet)
# sys.exit()
#torch.save(neuralnet,'./model.pth')
#torch.save(neuralnet.state_dict(),'./model_para.pth')
#MODEL_NAME = "Agooglenet_"
#MODEL_NAME = "Amnasnet10_"
#MODEL_NAME = "Ainception_v3_10_"
#MODEL_NAME = "Adensenet201_0_"
#MODEL_NAME = "Agooglenet_10_"
#MODEL_NAME = "AVGG19_300_"
#MODEL_NAME = "AVGG16_1062_"
#MODEL_NAME = "Ashufflenetv2x10_0_"
#MODEL_NAME = "den121_"
MODEL_NAME = "incepv3_"
AVERAGE_MODEL_TIME = 0
AVERAGE_MODEL_LAYER_TIME = np.zeros(torch.autograd.profiler.LAYER_NUM,dtype = np.float)
AVERAGE_MODEL_CONVLAYER_TIME = np.zeros(torch.autograd.profiler.CONVLAYER_NUM,dtype = np.float)

AVERAGE_MODEL_LAYER_TIME_P = np.zeros(torch.autograd.profiler.LAYER_NUM,dtype = np.float)
AVERAGE_MODEL_CONVLAYER_TIME_P = np.zeros(torch.autograd.profiler.CONVLAYER_NUM,dtype = np.float)

# Dump network architecture
#neuralnet.features[23]= nn.Conv2d(0, 0, kernel_size=(0, 0), stride=(1, 1), padding=(1, 1))
#neuralnet.features[24]= nn.Conv2d(0, 0, kernel_size=(0, 0), stride=(1, 1), padding=(1, 1))
#print(neuralnet)

# Remove the last fully-connected layer
#new_classifier = torch.nn.Sequential(*list(neuralnet.classifier.children())[:-7])
#neuralnet.classifier = new_classifier

# Dump network architecture
#neuralnet.features[23] = nn.ReLU(inplace=False)
#print(neuralnet)
#sys.exit()
# Use CPU for the inference
neuralnet.cpu()
# Set the model in the evaluation mode
neuralnet.eval()

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

#IMAGE_PATH = './data/tiny-imagenet-200/train/n01443537/images'
IMAGE_PATH = './data/ILSVRC2012_val'
#IMAGE_PATH = './data/test'
count = 0
counter = 0
layer_times = {}
Right_num = 0
total_times = 0.0

image_names = os.listdir(IMAGE_PATH)
image_names.sort()
#for img_file in os.listdir(IMAGE_PATH)[0:Input_N]:Input_N
for img_file in image_names[0:Input_N]:
    img = Image.open(IMAGE_PATH + '/' + img_file)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.autograd.profiler.profile(neuralnet) as prof:
        out = neuralnet(batch_t)
    prof.table(header="VGG13",top_level_events_only=True)# self_cpu_time_total
    #np.save(MODEL_NAME+'CONV_INDEX',torch.autograd.profiler.CONV_INDEX)
    #print(torch.autograd.profiler.CONV_INDEX)
    #sys.exit()
    #print(torch.autograd.profiler.CONV_INDEX)
    
    
    #sys.exit()
    prof.table(header=MODEL_NAME,top_level_events_only=True)
    #sys.exit()
    #print(global_var.model_count)
    #print(global_var.model_cpu_time)
    #print(prof.table())
    #print(prof.display(show_events = False))
    #erformance_ = str(prof.display(show_events = True))
    #with open("performance.csv","w") as f:
    torch.autograd.profiler.INPUT_INDEX =  torch.autograd.profiler.INPUT_INDEX + 1
    print(torch.autograd.profiler.INPUT_INDEX )

for m in range(1000):
    torch.autograd.profiler.MODEL_TIME[m] = torch.autograd.profiler.MODEL_TIME[m]/1000# ms
#     
# for m in range(10):
#     torch.autograd.profiler.MODEL_CONVLAYER_TIME[m] = torch.autograd.profiler.MODEL_CONVLAYER_TIME[m]/1000


    
    
# =============================================================================
# Time consumption of models
# =============================================================================

# print(torch.autograd.profiler.MODEL_CONVLAYER_TIME[0])
# print(torch.autograd.profiler.MODEL_CONVLAYER_TIME[1])
# print(torch.autograd.profiler.MODEL_CONVLAYER_TIME[2])
   

AVERAGE_MODEL_TIME = np.sum(torch.autograd.profiler.MODEL_TIME)/Input_N

print(AVERAGE_MODEL_TIME)
sys.exit()
    
AVERAGE_MODEL_LAYER_TIME = np.sum(torch.autograd.profiler.MODEL_LAYER_TIME,axis = 0)

# print("AVERAGE_MODEL_LAYER_TIME",AVERAGE_MODEL_LAYER_TIME.shape)

# print("torch.autograd.profiler.MODEL_LAYER_TIME",torch.autograd.profiler.MODEL_LAYER_TIME.shape)



for i in range(len(AVERAGE_MODEL_LAYER_TIME)):
    AVERAGE_MODEL_LAYER_TIME[i] = AVERAGE_MODEL_LAYER_TIME[i]/Input_N
AVERAGE_MODEL_CONVLAYER_TIME = np.sum(torch.autograd.profiler.MODEL_CONVLAYER_TIME,axis = 0)
for i in range(len(AVERAGE_MODEL_CONVLAYER_TIME)):
    AVERAGE_MODEL_CONVLAYER_TIME[i] = AVERAGE_MODEL_CONVLAYER_TIME[i]/Input_N
    
    
AVERAGE_MODEL_LAYER_TIME_P = np.sum(torch.autograd.profiler.MODEL_LAYER_TIME_P,axis = 0)
for i in range(len(AVERAGE_MODEL_LAYER_TIME_P)):
    AVERAGE_MODEL_LAYER_TIME_P[i] = AVERAGE_MODEL_LAYER_TIME_P[i]/Input_N
AVERAGE_MODEL_CONVLAYER_TIME_P = np.sum(torch.autograd.profiler.MODEL_CONVLAYER_TIME_P,axis = 0)
for i in range(len(AVERAGE_MODEL_CONVLAYER_TIME_P)):
    AVERAGE_MODEL_CONVLAYER_TIME_P[i] = AVERAGE_MODEL_CONVLAYER_TIME_P[i]/Input_N
    
# =============================================================================
# to save the model model_layer model_convlayer data
# =============================================================================
# np.save(MODEL_NAME+'1000_MODEL_TIME',torch.autograd.profiler.MODEL_TIME) # ms
# np.save(MODEL_NAME+'1000_MODEL_LAYER_TIME',torch.autograd.profiler.MODEL_LAYER_TIME)
# np.save(MODEL_NAME+'1000_MODEL_CONVLAYER_TIME',torch.autograd.profiler.MODEL_CONVLAYER_TIME)

#np.save(MODEL_NAME+'AVERAGE_MODEL_TIME',AVERAGE_MODEL_TIME)
np.save(MODEL_NAME+'AVERAGE_MODEL_LAYER_TIME',AVERAGE_MODEL_LAYER_TIME)
np.save(MODEL_NAME+'AVERAGE_MODEL_CONVLAYER_TIME',AVERAGE_MODEL_CONVLAYER_TIME)


# np.save(MODEL_NAME+'1000_MODEL_LAYER_TIME_P',torch.autograd.profiler.MODEL_LAYER_TIME_P)
# np.save(MODEL_NAME+'1000_MODEL_CONVLAYER_TIME_P',torch.autograd.profiler.MODEL_CONVLAYER_TIME_P)

# #np.save(MODEL_NAME+'AVERAGE_MODEL_TIME',AVERAGE_MODEL_TIME)
# np.save(MODEL_NAME+'AVERAGE_MODEL_LAYER_TIME_P',AVERAGE_MODEL_LAYER_TIME_P)
# np.save(MODEL_NAME+'AVERAGE_MODEL_CONVLAYER_TIME_P',AVERAGE_MODEL_CONVLAYER_TIME_P)

sys.exit()



# plt.tick_params(labelsize = 8) #

# x = range(100)
# plt.bar(x,torch.autograd.profiler.MODEL_TIME[0:100])
# plt.title("model_time")
# plt.xlabel("model")
# plt.ylabel("percentage")
# x_locator = plt.MultipleLocator(5)
# #y_locator = plt.MultipleLocator(20)
# axis = plt.gca()
# axis.xaxis.set_major_locator(x_locator)
# #axis.yaxis.set_major_locator(y_locator)
# plt.xlim(-1,100)

# plt.savefig("./model_time.eps",format = "eps",dpi = 1000)

# plt.clf()

# # =============================================================================
# # The data of each layer in the model
# # =============================================================================
# x =range(1,len(AVERAGE_MODEL_LAYER_TIME)+1)
# plt.bar(x,AVERAGE_MODEL_LAYER_TIME_P[0:len(AVERAGE_MODEL_LAYER_TIME)])
# plt.title("model_layer_time")
# plt.xlabel("layer")
# plt.ylabel("percentage")
# x_locator = plt.MultipleLocator(1)
# y_locator = plt.MultipleLocator(0.05)
# axis = plt.gca()
# axis.xaxis.set_major_locator(x_locator)
# axis.yaxis.set_major_locator(y_locator)
# plt.xlim(0,len(AVERAGE_MODEL_LAYER_TIME)+1)
# plt.ylim(0,0.2)
# plt.savefig("./model_layer.eps",format = "eps",dpi = 1000)
# plt.clf()

# # =============================================================================
# # The data of each convolution layer in the model
# # =============================================================================
# x = range(len(AVERAGE_MODEL_CONVLAYER_TIME))
# x = ["conv0","conv1","conv2","conv3","conv4","conv5","conv6","conv7","conv8","conv9"]
# p = plt.bar(x,AVERAGE_MODEL_CONVLAYER_TIME_P[0:len(AVERAGE_MODEL_CONVLAYER_TIME)])
# #print(torch.autograd.profiler.MODEL_CONVLAYER_TIME[0])
# plt.title("MODEL_CONVLAYER_TIME")
# plt.xlabel("convlayer")
# plt.ylabel("percentage")
# x_locator = plt.MultipleLocator(1)
# y_locator = plt.MultipleLocator(0.05)
# axis = plt.gca()
# axis.xaxis.set_major_locator(x_locator)
# #axis.yaxis.set_major_locator(y_locator)
# plt.xlim(-1,len(AVERAGE_MODEL_CONVLAYER_TIME))
# #plt.ylim(0,1)
# plt.savefig("./model_convlayer_time.eps",format = "eps",dpi = 1000)
# plt.clf()
#plt.show()

#print(AVERAGE_MODEL_TIME)
# print("sum_model_time",torch.autograd.profiler.SUM_MODLE_TIME)
# print("AVERAGE_MODEL_TIME:",torch.autograd.profiler.SUM_MODLE_TIME // Input_N)
# print(torch.autograd.profiler.MODEL_LAYER_LIST)
# print(torch.autograd.profiler.MODEL_CONVLAYER_TIME)
# print(torch.autograd.profiler.MODEL_LAYER_TIME)
# a = 0
# for i in range(50):
#     a = a + torch.autograd.profiler.MODEL_LAYER_TIME[i]
# print(a)




