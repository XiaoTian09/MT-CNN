# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:15:35 2018

@author: TX
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio 
#numpy.random.seed(7)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

matfn='predata_cnn.mat'
data=sio.loadmat(matfn)
x_pre=data['predata_cnn']
img_rows, img_cols = 1, 701
x_pre = x_pre.reshape(x_pre.shape[0], img_cols, 1)

model = load_model('single_polarity.cnn') 
 
classes=model.predict(x_pre,batch_size=8,verbose=0)
#np.savetxt("predict_singletrace1.txt", classes)
np.savetxt("predict_singletrace2.txt", classes)



