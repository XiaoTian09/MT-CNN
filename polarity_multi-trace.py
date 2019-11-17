'''
Predict the P-wave first motion polarity 
with multi-trace information using CNN.


'''

from __future__ import print_function
import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import matplotlib.pyplot as plt
import operator
from functools import reduce
#import loaddata
import numpy
import scipy.io as sio 
numpy.random.seed(7)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import time
time_start = time.time()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        

matfn='x_test.mat'
data=sio.loadmat(matfn)
x_test=data['x_test']
matfn='y_test.mat'
data=sio.loadmat(matfn)
y_test=data['y_test']
matfn='x_train.mat'
data=sio.loadmat(matfn)
x_train=data['x_train']
matfn='y_train.mat'
data=sio.loadmat(matfn)
y_train=data['y_train']


batch_size = 8
num_classes = 10
epochs = 200

# input image dimensions
img_rows, img_cols =num_classes, 701

# the data, shuffled and split between train and test sets
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 4),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2, 4),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(256, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='tanh'))
'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.SGD(lr=0.01, clipvalue=0.5),
              metrics=['accuracy'])
'''
model.compile(loss='mean_squared_error', 
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['mse'])

#history
history = LossHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)

model.save('polarity10_3.cnn')
weights=model.layers[0].get_weights()
#plot_model(model,to_file='test.png')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#plot acc-loss
history.loss_plot('epoch')


loss1=history.losses['epoch']+history.val_loss['epoch']
numpy.savetxt("loss_syn10_3.txt",loss1)

time_end = time.time()   

time_c= time_end - time_start   
print('time cost', time_c, 's')

