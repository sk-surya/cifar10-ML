#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
import readInput   # Local file
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import pandas as pd
import sys
import pickle
import os
import tensorflow as tf

# In[2]:
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
#print(os.getcwd())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

def getConvFeatures(img_arr):
    vgg16 = VGG16(input_shape=(32, 32, 3), weights='imagenet', include_top=False)
    conv_block_5_out = vgg16.get_layer('block5_pool').output
    feature_extractor_512b = Model(inputs=vgg16.input, outputs=conv_block_5_out)
    
    return np.squeeze(feature_extractor_512b.predict(img_arr))

def trainRandomForestClassifier():
    t1 = time.time()
    print('Training Random Forest Classifier')
    clf = RandomForestClassifier(n_estimators = 300, n_jobs=-1)
    
    train_img_arr, y_train = readInput.readTrainData()
    X_train = getConvFeatures(train_img_arr)
    clf.fit(X_train, y_train)
    print(f'Took {time.time()-t1} seconds.')
    print('Dumping classifier to file.')
    try:
        with open('vggRfcModel', 'wb') as writeFile:
            pickle.dump(clf, writeFile)
    except:
        print('Dumping model into file Failed.')
    finally:
        return clf

def predictTestData(clf):
    t1 = time.time()
    test_img_arr, y_test = readInput.readTestData()
    X_test = getConvFeatures(test_img_arr)
    predictions = clf.predict(X_test)
    accuracy = (y_test == predictions).mean()
    print(f'Test Accuracy = {accuracy*100} %.')
    print(f'Predicted in {time.time()-t1} seconds.')
    pred_df = pd.DataFrame(data={'Predictions' : predictions, 'Actual' : y_test})
    return pred_df

def saveDfToCsv(df):
    df.to_csv('mod_1_PredictionsRFC.csv')
    print('Saved Predictions.csv')


# In[3]:


def getClassifier(arg):
    if arg.lower() == 'forcetrain':
        clf = trainRandomForestClassifier()
    else:
        try:
            print('opening pre-trained model file.')
            with open(r"vggRfcModel", "rb") as inputFile:
                clf = pickle.load(inputFile)
            print('Loaded Pre-trained RFC classifier.')
        except FileNotFoundError:
            print('File not found. Initiate Force Training.')
            clf = trainRandomForestClassifier()
        finally:
            return clf
    return clf


# In[4]:


def main(arg):
    # Loading VGG16 model and defining feature extractor

    #arg = 'forcetrain' #sys.argv[1]
    #arg = 'dummy'
    clf_rfc = getClassifier(arg)
    df = predictTestData(clf_rfc)
    saveDfToCsv(df)
    
    print('Program Exited succesfully.')


# In[7]:


#main('forceTrain')
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