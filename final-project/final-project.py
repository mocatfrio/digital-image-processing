#!/usr/bin/env python
# coding: utf-8

# ### Final Project
# # Classification of Clothing Motifs
# 
# --------------------
# ```
# Hafara Firdausi (05111950010040)
# Digital Image Processing
# ```
# 
# ## 1. Description
# ### 1.1 Purpose
# 
# Automatically classifies clothes based on their motif rather than manually input categories in the online shop.
# 
# ### 1.2 Methodology
# 
# ![](method.png)
# 

# ## 2. Steps
# ### 2.1 Import Libraries

# In[1]:


# import required libraries

import numpy as np # for numerical computations
import pandas as pd # for dataframe operations

from matplotlib import pyplot as plt #for viewing images and plots
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2   #For image processing

from sklearn.preprocessing import LabelEncoder       #For encoding categorical variables
from sklearn.model_selection import train_test_split #For splitting of dataset

#All tensorflow utilities for creating, training and working with a CNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


# ### 2.2 Import and Prepare Dataset
# The dataset used is Fashion Dataset from **Kaggle**, containing **15000 images with various dress categories**. All images from real fashion photos. So, there is a lot of background noise. But the dresses in the images have been **tagged over by red rectangles**.

# In[2]:


# define dataset file
# I only use 10,000 data
dataset = "dress-10k.csv"

# import the dataset
df = pd.read_csv(dataset)
df.head(10)


# In[3]:


len(df)


# In[4]:


# download all images

# import wget
# import os
# import pandas as pd #for dataframe operations

# # define 
# dataset_dir = "dress-10k.csv"
# img_dir = "img"

# # import the dataset
# df = pd.read_csv(dataset_dir)
# df.head(10)

# # make directory
# if not os.path.exists(img_dir):
#     os.makedirs(img_dir)
    
# for url in df['image_url'] :
#     local_file = wget.download(url, img_dir)
#     print(local_file)


# In[5]:


# convert image url to image path
img_path = []
img_dir = "img/"

for url in df['image_url'] :
    new_path = img_dir + url.split('/')[-1]
    img_path.append(new_path)

df['img_path'] = img_path


# In[6]:


df.head(10)


# In[7]:


# drop "image_url" column
df.drop("image_url", axis=1, inplace=True)


# In[8]:


df.head(10)


# In[9]:


# display some images
fig = plt.figure(figsize=(15, 15))
columns = 3
rows = 3
for i in range(1, columns*rows +1):
    img = cv2.imread(df['img_path'].loc[i])[:,:,::-1]
    fig.add_subplot(rows, columns, i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img)
plt.show()


# In[10]:


# list unique categories
print('All categories : \n ')
for category in df['category'].unique():
    print(category)
    
print('\n ')

# total of unique categories
n_classes = df['category'].nunique()
print('Total number of unique categories:', n_classes)


# In[11]:


# remove the category 'OTHER' from the dataset
df = df.loc[(df['category'] != 'OTHER')].reset_index(drop=True)


# ### 2.3 Preprocess Image
# ### 2.3.1 Masking
# **Image masking** is the process of separating an image from its background, either to cause the image to stand out on its own or to place the image over another background. This process used to **separate red rectangles** from the whole image.

# In[12]:


test_img = df['img_path'].loc[2]

# original image
image = cv2.imread(test_img)

# convert to HSV for creating a mask
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# create a mask that detects the red rectangular tags present in each image
mask = cv2.inRange(image_hsv, (0,255,255), (0,255,255))

plt.figure(figsize=(15, 15))
plt.subplot(1,3,1), plt.imshow(image),plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(image_hsv),plt.title('Image HSV')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(mask),plt.title('Mask')
plt.xticks([]), plt.yticks([])
plt.show()


# In[13]:


# get the coordinates of the red rectangle in the image

if len(np.where(mask != 0)[0]) != 0:
    y1 = min(np.where(mask != 0)[0])
    y2 = max(np.where(mask != 0)[0])
else:
    y1 = 0                                     
    y2 = len(mask)

if len(np.where(mask != 0)[1]) != 0:
    x1 = min(np.where(mask != 0)[1])
    x2 = max(np.where(mask != 0)[1])
else:
    x1 = 0
    x2 = len(mask[0])
    
print("y1 : {}\ny2 : {}\nx1 : {}\nx2 : {}".format(y1, y2, x1, x2))    


# ### 2.3.2 Median Filtering (Image Enhancement)
# **Median Filter** is a non-linear digital filtering technique, often used to remove noise from an image or signal. Such noise reduction is a typical pre-processing step to improve the results of later processing (for example, edge detection on an image).

# In[14]:


# median filtering
# the dimension of the x and y axis of the kernal.
figure_size = 3
image_enhanced = cv2.medianBlur(image_hsv, figure_size)

plt.figure(figsize=(15, 15))
plt.subplot(1,2,1), plt.imshow(image_hsv),plt.title('Image HSV')
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(image_enhanced),plt.title('Median Filtering')
plt.xticks([]), plt.yticks([])
plt.show()


# ### 2.3.3 Cropping
# After get the coordinates of the red rectangle in the image and apply median filtering, the image is cropped based on those coordinates.

# In[15]:


# convert the filtered image back to BGR
image_enhanced_bgr = cv2.cvtColor(image_enhanced, cv2.COLOR_HSV2BGR)

# convert to grayscale that will actually be used for training
image_gray = cv2.cvtColor(image_enhanced_bgr, cv2.COLOR_BGR2GRAY)

# crop the grayscle image along those coordinates
image_cropped = image_gray[y1:y2, x1:x2]

plt.figure(figsize=(15, 15))
plt.subplot(1,3,1), plt.imshow(image_enhanced_bgr),plt.title('Image HSV Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(image_gray),plt.title('Image HSV Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(image_cropped),plt.title('Image Cropped')
plt.xticks([]), plt.yticks([])
plt.show()


# ### 2.3.4 Resizing
# Resize cropped image to 100x100 pixels size.

# In[16]:


# resize the image to 100x100 pixels size
image_100x100 = cv2.resize(image_cropped, (100, 100))
plt.imshow(image_100x100)


# In[17]:


# save image as in form of array of 10000x1
image_arr = image_100x100.flatten()
print(image_arr)
image_arr.shape


# ### 2.3.5 Preprocess All Data
# After that, doing preprocess to all data.

# In[18]:


def preprocess(img_path):
    # original image
    image = cv2.imread(img_path)

    # convert to HSV for creating a mask
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # create a mask that detects the red rectangular tags present in each image
    mask = cv2.inRange(image_hsv, (0,255,255), (0,255,255))
    
    # get the coordinates of the red rectangle in the image
    if len(np.where(mask != 0)[0]) != 0:
        y1 = min(np.where(mask != 0)[0])
        y2 = max(np.where(mask != 0)[0])
    else:
        y1 = 0                                     
        y2 = len(mask)

    if len(np.where(mask != 0)[1]) != 0:
        x1 = min(np.where(mask != 0)[1])
        x2 = max(np.where(mask != 0)[1])
    else:
        x1 = 0
        x2 = len(mask[0])
        
    # median filtering
    # the dimension of the x and y axis of the kernal.
    figure_size = 3
    image_enhanced = cv2.medianBlur(image_hsv, figure_size)
    
    # convert the filtered image back to BGR
    image_enhanced_bgr = cv2.cvtColor(image_enhanced, cv2.COLOR_HSV2BGR)

    # convert to grayscale that will actually be used for training
    image_gray = cv2.cvtColor(image_enhanced_bgr, cv2.COLOR_BGR2GRAY)

    # crop the grayscle image along those coordinates
    image_cropped = image_gray[y1:y2, x1:x2]
    
    # resize the image to 100x100 pixels size
    image_100x100 = cv2.resize(image_cropped, (100, 100))
    
    # save image as in form of array of 10000x1
    image_arr = image_100x100.flatten()
    return image_arr


# In[19]:


preprocessed_img = []

for img_path in df['img_path'] :
    preprocessed_img.append(preprocess(img_path))

X = np.array(preprocessed_img)

print(X)
X.shape


# In[20]:


# display some preprocessed images
fig = plt.figure(figsize=(15, 15))
columns = 3
rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(preprocessed_img[i].reshape(100, 100)), plt.axis('off')
    plt.xticks([]), plt.yticks([])
plt.show()


# ### 2.4 Split Data
# Split data into train, test, and validation set.

# In[21]:


# creating target (Y)
# tranform category label into to numerical labels

encoder = LabelEncoder()
Targets = encoder.fit_transform(df['category'])
Targets
Targets.shape


# In[22]:


# one-hot encoding 
Y = to_categorical(Targets, num_classes = n_classes)
Y[0:3]
Y.shape


# In[23]:


# segregation of a test set for testing on the trained model
X_test = X[8000:,]
Y_test = Y[8000:,]

# separation of a validation set from the remaing training set (required for validation while training)
X_train, X_val, Y_train, Y_val = train_test_split(X[:8000,], Y[:8000,], test_size=0.15, random_state=13)


# In[24]:


# reshape the input matrices such that each sample is three-dimensional

img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# ### 2.5 Classification using Convolutional Neural Network (CNN)
# ### 2.5.1 Train Model

# In[25]:


# define the CNN Model

model = Sequential()

# 16 Convolutional Layer
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

# Max Pooling Layer
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

# 32 Convolution Layer
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

# Max Pooling Layer
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(n_classes, activation='softmax'))

learning_rate = 0.001

model.compile(loss = categorical_crossentropy,
              optimizer = Adam(learning_rate),
              metrics=['accuracy'])

model.summary()


# In[26]:


# saving the best weight during training 

save_at = "model.hdf5"
save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')


# In[27]:


# train the CNN model

history = model.fit(X_train, Y_train, 
                    epochs = 30, batch_size = 100, 
                    callbacks=[save_best], verbose=1, 
                    validation_data = (X_val, Y_val))


# In[28]:


# plot accuracy
plt.figure(figsize=(7, 5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy', weight='bold', fontsize=16)
plt.ylabel('Accuracy', weight='bold', fontsize=14)
plt.xlabel('Epoch', weight='bold', fontsize=14)
plt.legend(['Train', 'Validation'], loc='upper left', prop={'size': 14})


# ### 2.5.2 Evaluating Performace over Test-set

# In[29]:


# run model on the held-out test set

# model = load_model('model.hdf5')
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


# In[30]:


Y_pred = np.round(model.predict(X_test))

np.random.seed(87)
for rand_num in np.random.randint(0, len(Y_test), 5):
    plt.figure()
    plt.imshow(X_test[rand_num].reshape(100, 100)), plt.axis('off')
    if np.where(Y_pred[rand_num] == 1)[0].sum() == np.where(Y_test[rand_num] == 1)[0].sum():
        plt.title(encoder.classes_[np.where(Y_pred[rand_num] == 1)[0].sum()], color='g')
    else :
        plt.title(encoder.classes_[np.where(Y_pred[rand_num] == 1)[0].sum()], color='r')


# ----
# I want to compare the accuration of preprocessed images with unpreprocessed ones.

# In[31]:


img_list = []

for img_path in df['img_path'] :
    # original image
    image = cv2.imread(img_path)
    
    # convert to grayscale that will actually be used for training
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # resize the image to 100x100 pixels size
    image_100x100 = cv2.resize(image_gray, (100, 100))
    
    # save image as in form of array of 10000x1
    image_arr = image_100x100.flatten()
    
    img_list.append(image_arr)

X = np.array(img_list)

print(X)
X.shape


# In[32]:


# display some images
fig = plt.figure(figsize=(15, 15))
columns = 3
rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_list[i].reshape(100, 100)), plt.axis('off')
    plt.xticks([]), plt.yticks([])
plt.show()


# ### 2.4 Split Data

# In[33]:


# creating target (Y)
# tranform category label into to numerical labels

encoder = LabelEncoder()
Targets = encoder.fit_transform(df['category'])
Targets
Targets.shape


# In[34]:


# one-hot encoding 
Y = to_categorical(Targets, num_classes = n_classes)
Y[0:3]
Y.shape


# In[35]:


# segregation of a test set for testing on the trained model
X_test = X[8000:,]
Y_test = Y[8000:,]

# separation of a validation set from the remaing training set (required for validation while training)
X_train, X_val, Y_train, Y_val = train_test_split(X[:8000,], Y[:8000,], test_size=0.15, random_state=13)


# In[36]:


# reshape the input matrices such that each sample is three-dimensional

img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


# ### 2.5 Classification using Convolutional Neural Network (CNN)
# ### 2.5.1 Train Model

# In[37]:


# define the CNN Model

model = Sequential()

# 16 Convolutional Layer
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

# Max Pooling Layer
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

# 32 Convolution Layer
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

# Max Pooling Layer
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(n_classes, activation='softmax'))

learning_rate = 0.001

model.compile(loss = categorical_crossentropy,
              optimizer = Adam(learning_rate),
              metrics=['accuracy'])

model.summary()


# In[38]:


# saving the best weight during training 

save_at = "model.hdf5"
save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')


# In[39]:


# train the CNN model

history = model.fit(X_train, Y_train, 
                    epochs = 15, batch_size = 100, 
                    callbacks=[save_best], verbose=1, 
                    validation_data = (X_val, Y_val))


# In[40]:


# plot accuracy
plt.figure(figsize=(7, 5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy', weight='bold', fontsize=16)
plt.ylabel('Accuracy', weight='bold', fontsize=14)
plt.xlabel('Epoch', weight='bold', fontsize=14)
plt.legend(['Train', 'Validation'], loc='upper left', prop={'size': 14})


# ### 2.5.2 Evaluating Performace over Test-set

# In[41]:


# run model on the held-out test set

# model = load_model('model.hdf5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


# In[42]:


Y_pred = np.round(model.predict(X_test))

np.random.seed(87)
for rand_num in np.random.randint(0, len(Y_test), 5):
    plt.figure()
    plt.imshow(X_test[rand_num].reshape(100, 100)), plt.axis('off')
    if np.where(Y_pred[rand_num] == 1)[0].sum() == np.where(Y_test[rand_num] == 1)[0].sum():
        plt.title(encoder.classes_[np.where(Y_pred[rand_num] == 1)[0].sum()], color='g')
    else :
        plt.title(encoder.classes_[np.where(Y_pred[rand_num] == 1)[0].sum()], color='r')


# ## 3. Discussion and Conclusion
# 
# Based on the evaluation results above, **the preprocessed image gets higher accuracy compared to unpreprocessed image**, with same parameters, that is:
# * Split data:
#     * Train set: **6800**
#     * Test set: **1669**
#     * Validation set: **1200**
# * Epoch: **15**
# * Batch Size: **100**
# 
# 
# | Scenario | Accuracy | 
# |---|---|
# |With preprocess | 58.36 % |
# |Without preprocess | 56.32 % |
# 
# Based on the plot accuracy graphs, the model has **high training accuracy and very low validation**. This case is probably known as **overfitting**. Overfitting is such a problem because the evaluation of machine learning algorithms on training data is different from the evaluation we actually care the most about, namely how well the algorithm performs on unseen data.
# 
# There are two important techniques that you can use when evaluating machine learning algorithms to limit overfitting:
# 
# 1. Use a resampling technique to estimate model accuracy.
# 2. Hold back a validation dataset.
# 
# The most popular resampling technique is **k-fold cross validation**. It allows you to train and test your model k-times on different subsets of training data and build up an estimate of the performance of a machine learning model on unseen data.
# 
# A validation dataset is simply a subset of your training data that you hold back from your machine learning algorithms until the very end of your project. After you have selected and tuned your machine learning algorithms on your training dataset you can evaluate the learned models on the validation dataset to get a final objective idea of how the models might perform on unseen data.

# ## References
# * https://www.kaggle.com/nitinsss/fashion-dataset-with-over-15000-labelled-images
# * https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/
