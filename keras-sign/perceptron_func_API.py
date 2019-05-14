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

# create model
inp = Input(shape=(img_width, img_height))
reshape_out = Reshape((img_width, img_height, 1))(inp)
conv2D_1_out = Conv2D(64, (3,3), padding='same', activation="relu")(reshape_out)
maxpool2D_1_out = MaxPooling2D((2,2))(conv2D_1_out)
conv2D_2_out = Conv2D(64, (3,3), padding='same')(reshape_out)
conv2D_2_act_out = LeakyReLU(0.1)(conv2D_2_out)
maxpool2D_2_out = MaxPooling2D((2,2))(conv2D_2_act_out)
conc_conv2D_out = Concatenate()([maxpool2D_1_out, maxpool2D_2_out])
batchnorm1_out = BatchNormalization()(conc_conv2D_out)
flat1_out = Flatten()(batchnorm1_out)
dense1_out = Dense(128, activation="relu")(flat1_out)
dropout1_out = Dropout(0.25)(dense1_out)
dense2_out = Dense(64, activation="relu")(dropout1_out)
dropout2_out = Dropout(0.2)(dense2_out)
dense3_out = Dense(num_classes, activation="softmax")(dropout2_out)
model = Model(inp, dense3_out)

#model = Sequential()
#model.add(Reshape((img_width, img_height, 1), input_shape=(img_width, img_height)))
#model.add(Conv2D(64, (3,3), padding='same', activation="relu"))
#model.add(MaxPooling2D((2,2)))
#model.add(Conv2D(32, (3,3), padding='same', activation="relu"))
#model.add(MaxPooling2D((2,2)))
#model.add(BatchNormalization())
#model.add(Flatten())
#model.add(Dense(128, activation="relu"))
#model.add(Dropout(0.25))
#model.add(Dense(64, activation="relu"))
#model.add(Dropout(0.2))
#model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=config.loss, optimizer=config.optimizer,
              metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
