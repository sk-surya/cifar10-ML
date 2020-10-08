#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np


# In[2]:


# In[4]:


def readBinFile(filename):
    data = bytes()
    with open(filename, mode='rb') as file:
        data += file.read()
    return data

def getDatafromBin(filenames):
    data = bytes()
    
    for filename in filenames:
        data = data + readBinFile(filename)
    rgb = np.frombuffer(data, dtype=np.uint8)
    num_img = int(len(rgb)/3073)
    rgb2 = rgb.reshape(num_img, 3073)
    labels = rgb2[:, 0]
    features = rgb2[:, 1:]
    rgb_3d = features.reshape(num_img, 3, 1024)
    img_arr = rgb_3d.swapaxes(1,2).reshape(len(rgb_3d), 32, 32, 3)
    return img_arr, labels

def readClassnames():
    with open('./dataset/' + 'batches.meta.txt', mode='r') as file:
        txt_content = file.read()
    class_names = [x for x in txt_content.splitlines() if len(x.strip()) > 0]
    return class_names

def readTrainData():
    trainFilenames = ["./dataset/" + x for x in os.listdir('dataset') if '.bin' in x and 'batch' in x and 'test' not in x]
    assert trainFilenames == [   './dataset/data_batch_1.bin',
                                 './dataset/data_batch_2.bin',
                                 './dataset/data_batch_3.bin',
                                 './dataset/data_batch_4.bin',
                                 './dataset/data_batch_5.bin'], 'Training Filenames not found as expected'
    return getDatafromBin(trainFilenames)

def readTestData():
    testFilenames = ["./dataset/test_batch.bin"]
    return getDatafromBin(testFilenames)


# In[7]:


# train_img_arr, trainFeatures = readTrainData()
# test_img_arr, testFeatures = readTestData()


# In[ ]:




