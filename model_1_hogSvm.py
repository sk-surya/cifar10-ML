#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2    # pip install opencv-python
import readInput    # this is not a package, this is a local file (located in the folder)
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import time
import pickle
import os
import sys

# In[2]:
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def create_SVM():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(0.1)
    svm.setGamma(0.1)
    return svm

def getHogDescriptors(img_arr):
    # HOG Parameters:
    winSize = 32
    blockSize = 12
    blockStride = 4
    cellSize = 4
    nbins = 18
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride,blockStride),(cellSize,cellSize),nbins,derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)
    return np.array([hog.compute(img).flatten() for img in img_arr])
    
def getPCA(X_train):
    t1 = time.time()
    pca = PCA(3000)
    pca.fit(X_train)
    print('dumping pca file.')
    with open('SvmHogPca', 'wb') as writeFile:
        pickle.dump(pca, writeFile)
    print(f'pca took {time.time()-t1} seconds.')
    return pca

def trainHogSvmClassifier():
    print('Training HOG + SVM Classifier')
    svm = create_SVM()
    
    train_img_arr, y_train = readInput.readTrainData()
    X_train = getHogDescriptors(train_img_arr)
    
    pca = getPCA(X_train)
    
    X_train = pca.transform(X_train)
    
    t2 = time.time()
    svm.train(np.asarray(X_train), cv2.ml.ROW_SAMPLE, np.asarray(y_train, dtype=np.int32))
    print(f'SVM training took {time.time()-t2} seconds')
    print('Dumping classifier to file.')
    svm.save('SvmHogModel')
    return svm, pca

def predictTestData(svm, pca):
    t1 = time.time()
    test_img_arr, y_test = readInput.readTestData()
    
    X_test = getHogDescriptors(test_img_arr)
    X_test = pca.transform(X_test)
    
    predictions = svm.predict(np.asarray(X_test))[1].ravel()
    accuracy = (y_test == predictions).mean()
    print(f'Test Accuracy = {accuracy*100} %.')
    print(f'Predicted in {time.time()-t1} seconds.')
    pred_df = pd.DataFrame(data={'Predictions' : predictions, 'Actual' : y_test})
    return pred_df

def saveDfToCsv(df):
    df.to_csv('mod_2_PredictionsHogSvm.csv')
    print('Saved Predictions.csv')


# In[3]:


def getClassifier(arg):
    if arg.lower() == 'forcetrain':
        clf_svm, pca = trainHogSvmClassifier()
    else:
        try:
            print('opening pre-trained model file.')
            #with open(r"SvmHogModel", "rb") as inputFile:
            clf_svm = cv2.ml.SVM_load('SvmHogModel')
            with open(r"SvmHogPca", "rb") as inputFile:
                pca = pickle.load(inputFile)
            print('Loaded Pre-trained HogSVM classifier and pca.')
        except FileNotFoundError:
            print('File not found. Initiate Force Training.')
            clf_svm, pca = trainHogSvmClassifier()
    return clf_svm, pca


# In[4]:


def main(arg):
    # Loading VGG16 model and defining feature extractor

    #arg = 'forcetrain' #sys.argv[1]
    #arg = 'dummy'
    clf, pca = getClassifier(arg)
    df = predictTestData(clf, pca)
    saveDfToCsv(df)
    
    print('Program Exited succesfully.')


# In[6]:


arg_cnt = len(sys.argv)
if arg_cnt == 1:
    print("""Error: Argument missing. Please use 'forcetrain' or 'pretrain'""")
else:
    arg = sys.argv[1]
    if arg.lower() not in ['forcetrain', 'pretrain']:
        print("""Error: Incorrect argument. Please use 'forcetrain' or 'pretrain'""")
        exit(1)
    else:
        main(arg)
        exit(0)




