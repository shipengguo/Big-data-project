# -*-coding:UTF-8


import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D 
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from getdataset import get_dataset
from keras.layers.recurrent import LSTM, GRU


# load the dataset 
print('Loading data...')
X, y, X_test, test_id = get_dataset()
#x_total = np.hstacks((X,y))
x_train = X[0:1600,:]
y_train = y[0:1600,:]
x_vali = X[1600:2000,:]
y_vali = y[1600:2000,:]
print(len(x_train), 'train sequences')
print(len(x_vali), 'validation sequences')
print(len(X_test), 'test sequences')

model = Sequential()

conv_layer = Conv1D(filters=500,
                    kernel_size=2,
                    strides=1,
                    padding='valid',
                    activation='relu',
                    input_shape=(14,4))

model.add(conv_layer)
model.add(MaxPooling1D(pool_size=13,
                       strides=13))
model.add(Dropout(0.4))
#model.add(Bidirectional(LSTM(320, return_sequences=True)))
model.add(GRU(600, return_sequences=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(925,
                activation='relu'))
model.add(Dense(1,
                activation='sigmoid'))


model.compile(loss = 'binary_crossentropy' , optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size=200, epochs=500, verbose=1, validation_split=0.1,shuffle=True)

# Final evaluation of the model
score, acc = model.evaluate(x_vali, y_vali,
                            verbose=0)
print('validation score:', score)
print('validation accuracy:', acc)

# predict test dataset
preds = model.predict(X_test)
print 'prediction:\n', preds
for pred in preds:
	if pred[0] > 0.5:
		pred[0] = 1
	elif pred[0] < 0.5:
		pred[0] = 0
print 'prediction_new:\n ',preds

#plot training process
print(history.history.keys())