# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Input, Concatenate, LeakyReLU
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "mae"
config.optimizer = "adam"
config.epochs = 10

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()
#print(X_train.shape)
#print(X_test.shape)

#for i in range(26):
#    print(np.sum(y_train==i))

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
#print(y_train.shape)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(y_train.shape)
#print(np.histogram(y_train, bins=26))
    
num_classes = y_train.shape[1]
#print (num_classes)

# you may want to normalize the data here..

# normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

def inception_block(ip):
    conv1x1_out = Conv2D(1, (1,1), padding='same', activation="relu")(ip)
    conv3x3_out = Conv2D(1, (3,3), padding='same', activation="relu")(ip)
    conv5x5_out = Conv2D(1, (5,5), padding='same', activation="relu")(ip)
    maxpool3x3_out = MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(ip)
    concat_out = Concatenate()([conv1x1_out, conv3x3_out, conv5x5_out, maxpool3x3_out])
    return concat_out

# create model
inp = Input(shape=(img_width, img_height))
reshape_out = Reshape((img_width, img_height, 1))(inp)
incept1_out = inception_block(reshape_out)
maxpool1_out = MaxPooling2D((2,2))(incept1_out)
incept2_out = inception_block(maxpool1_out)
batchnorm1_out = BatchNormalization()(incept2_out)
flat1_out = Flatten()(batchnorm1_out)
dense1_out = Dense(128, activation="relu")(flat1_out)
dropout1_out = Dropout(0.25)(dense1_out)
dense2_out = Dense(64, activation="relu")(dropout1_out)
dropout2_out = Dropout(0.2)(dense2_out)
dense3_out = Dense(num_classes, activation="softmax")(dropout2_out)
model = Model(inp, dense3_out)

model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
