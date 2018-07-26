
# coding: utf-8

# In[1]:


# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback


# In[2]:


print(signdata.letters)


# In[3]:


# logging code
run = wandb.init()
config = run.config
config.team_name = "davros"
config.loss = "sparse_categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10


# In[4]:


# load data
(x_test, y_test) = signdata.load_test_data()
(x_train, y_train) = signdata.load_train_data()


# In[5]:


img_width = x_test.shape[1]
img_height = x_test.shape[2]

print(img_width, img_height)
input_shape = (img_width, img_height, 1)


# In[6]:


X_test = x_test /255.0
X_train = x_train /255.0


# In[7]:


num_classes = y_train.max()+1
print(num_classes)


# In[ ]:


# one hot encode outputs
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# In[8]:


# Reshape from (num_examples, x, y) to (num_examples, x, y, num_channels)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape, X_test.shape)


# In[9]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[10]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train,
      epochs=config.epochs,
      verbose=1,
      validation_data=(X_test, y_test),
      callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

