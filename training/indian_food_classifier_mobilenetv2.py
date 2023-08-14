#%% [markdown]
# Import necessary libraries
#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings("ignore")

#%% [markdown]
# Avoid OOM errors by setting GPU Memory Consumption Growth as True
#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


tf.test.is_gpu_available()

#%% [markdown]
#Get the raw data path from which we will make our structured dataset folders
#%%
# os.getcwd()
# raw_data_path = os.path.join(os.getcwd(), 'raw_dataset')

#%% [markdown]
# Create a structured dataset from raw dataset
#%%
# From the raw path we can make a list of all the categories or classes by listing the folder names
category_list = os.listdir(os.path.join(os.getcwd(), os.pardir), 'images', 'train'))

# Set the name of root directory of our final structured dataset
root_dir = 'images'

# # Just in case we delete a pre-existing folder for the root directory
# if os.path.exists(os.path.join(os.getcwd(), root_dir)):
#     shutil.rmtree(os.path.join(os.getcwd(), root_dir))

# # Create train validation and test folders inside root directory
# # Loop through each category in the list
# for i in category_list:
    
#     # For each category create a folder within train, validation and test folders respectively
#     if not os.path.exists(f'{root_dir}/train/{i}'):
#         os.makedirs(f'{root_dir}/train/{i}')
#     if not os.path.exists(f'{root_dir}/validation/{i}'):
#         os.makedirs(f'{root_dir}/validation/{i}')
#     if not os.path.exists(f'{root_dir}/test/{i}'):
#         os.makedirs(f'{root_dir}/test/{i}')

#     source = raw_data_path + '/' + i

#     allFileNames = os.listdir(source)

#     np.random.shuffle(allFileNames)

#     test_split_ratio = 0.3
#     test_len = int(len(allFileNames)*test_split_ratio)
#     train_FileNames = allFileNames[:-test_len]
#     temp_test_FileNames = allFileNames[-test_len:]

#     validation_split_ratio = 0.5
#     validation_len = int(len(temp_test_FileNames)*validation_split_ratio)

#     test_FileNames = temp_test_FileNames[:-validation_len]
#     validation_FileNames = temp_test_FileNames[-validation_len:]

#     for name in train_FileNames:
#         pass
#         shutil.copy(f'{source}/{name}', f'{root_dir}/train/{i}')

#     for name in test_FileNames:
#         shutil.copy(f'{source}/{name}', f'{root_dir}/test/{i}')

#     for name in validation_FileNames:
#         shutil.copy(f'{source}/{name}', f'{root_dir}/validation/{i}')


# %%
#Setting dataset path for train and test sets 
train_path = f"{root_dir}/train"
validation_path = f"{root_dir}/validation"
test_path = f"{root_dir}/test"

# %% [markdwon]
# Image Data Generator is used to augment existing images to create variations of the image
# By using this method we can generate a bigger dataset than the one we have by augmenting and creating new images from existing ones

#%%
batch_size = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True)


#%%
train_set = train_datagen.flow_from_directory(train_path,
                                              target_size = (224, 224),
                                              batch_size = 32,
                                              class_mode = 'categorical')


# validtion Data Augmentation 
validation_datagen = ImageDataGenerator(rescale=1./255)



validation_set = validation_datagen.flow_from_directory(validation_path,
                                                        target_size = (224, 224),
                                                        batch_size = 32,
                                                        class_mode = 'categorical')



#%% [markdown]
# Sample on batch from validation set for analysis and cross checking shapes
#%%
sample_batch = validation_set.next()

sample_images = sample_batch[0]
sample_labels = sample_batch[1]
sample_images.shape

# for i,j in zip(sample_images, sample_labels):
#     # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
#     cv2.imshow(f"{j}",i)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

#%% [markdown]
# Import TF Requirements
#%%
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

IMAGE_SIZE = [224, 224, 3]
base_model = MobileNetV2(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)

#%% [markdown]
# input one batch of our images into the base model and see how it is converted after passing through
# We can check the shape of the output from the last layer
# we will get extracted features once we pass an image through the base model

#%%
sample_images.shape

features_from_base_model = base_model(sample_images)
print(features_from_base_model.shape)

# %% [markdown]
# parameters of the Mobilenet V2 model are frozen to prevent them from being updated during training. 
# By freezing the model we make sure that it doesn't retrain the layers
# We want to retain whatever it has already learnt 

#%%
base_model.trainable = False
base_model.summary()

#%% [markdown]
# We are adding our own trainable layer on top of the base model 
# We can input the feature batch we retrieved from out previous layer to this and check the output shape

# %%
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

global_average_layer = GlobalAveragePooling2D()
features_from_average_layer = global_average_layer(features_from_base_model)

print(features_from_average_layer.shape)

#%% [markdown]
# We are adding the final softmax layer that will give us the prediction

#%%
num_classes = len(category_list)
prediction_layer = Dense(num_classes, activation='softmax')
features_from_prediction_layer = prediction_layer(features_from_average_layer)
print(features_from_prediction_layer.shape)

#%% [markdown]
# We form our final model using 
# Base Model
# the flattened  global average layer
# the final prediction dense layer

#%%


model = Sequential([
   base_model,
   global_average_layer,
   prediction_layer
])

model.summary()

#%% [markdown]
# We initialize our parameters for training such as the optimizer, loss and learning rates
#%%
adam = optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# %%
checkpoint = ModelCheckpoint(filepath='mobilenetv2.h5', 
                               verbose=2, 
                               save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()

history = model.fit(train_set,
                      validation_data=validation_set,
                      epochs=20,
                      validation_steps=len(validation_set),
                      callbacks=callbacks, 
                      verbose=2)


duration = datetime.now() - start
print("Training completed in time: ", duration)


#%% [markdown]
# Function to take training history and plot the accuracy and loss metrics
#%%

def plot_accuracy_loss(history):
   
   fig = plt.figure(figsize= (10,5))

   #Accuracy Plot
   plt.subplot(221)
   plt.plot(history.history['accuracy'], 'bo--', label= "acc")
   plt.plot(history.history['val_accuracy'], 'ro--', label= "val_acc")
   plt.title("Training Accuracy vs Validation Accuracy")
   plt.ylabel("accuracy")
   plt.xlabel("epochs")
   plt.legend()


   #Loss Function Plot
   plt.subplot(222)
   plt.plot(history.history['loss'], 'bo--', label= "loss")
   plt.plot(history.history['val_loss'], 'ro--', label= "val_loss")
   plt.title("Training Loss vs Validation Loss")
   plt.ylabel("loss")
   plt.xlabel("epochs")
   plt.legend()

   plt.show()


plot_accuracy_loss(history)


#%% [markdown]
# Function to pre process test images to be input into model for prediction
# %%
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def process_input_image(path):
    img = load_img(path, target_size= (224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    return img_array



#%% [markdown]
# Function to feed the path of test images to obtain a visual of the image and its predicted value
#%%
def test_predict_view(test_path, num_examples = 10):
    test_img_paths = list(test_path.glob(r'**/*.jpg'))

    test_labels = []

    for x in test_img_paths:
        test_labels.append(os.path.split(os.path.split(x)[0])[1])


    test_df = pd.DataFrame({"filepath" : test_img_paths, "label" : test_labels})
    test_df["filepath"] = test_df["filepath"].astype(str)


    test_df = test_df.sample(frac=1)
    test_df = test_df.iloc[:num_examples,:]

    for index, row in test_df.iterrows():
        current_img = cv2.imread(row['filepath'], cv2.COLOR_BGR2RGB)
        current_img = cv2.resize(current_img, (540,540))
        current_img_processed = process_input_image(row['filepath'])
        current_pred_array = model.predict(current_img_processed)
        current_pred = np.argmax(current_pred_array, axis=1)[0]
        predicted_food_item = category_list[current_pred]
        actual_food_item = row["label"]
        print(f"Predicted :  {predicted_food_item} Actual : {actual_food_item}", )
        cv2.imshow("Test Image",current_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

#%% [markdown]
# Call the visual test function on the trained model
#%%
test_path = Path(os.path.join(os.getcwd(), root_dir, 'test'))

test_predict_view(test_path)



#%% [markdown]
### FINE TUNING ###
## UNFREEZING MOBILENET V2 BASE LAYERS ###




# %%
# The base_model parameters are unfrozen to allow fine-tuning of the entire model
base_model.trainable = True
model.summary()

# %%
# Adam optimizer is created with a lower learning rate (1e-5) for fine-tuning.

#%%
adam = optimizers.Adam(1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#%%

checkpoint = ModelCheckpoint(filepath='mobilenetv2.h5', 
                               verbose=2, save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()

history_finetuned = model.fit(train_set,
                      validation_data=validation_set,
                      epochs=50,
                      validation_steps=len(validation_set),
                      callbacks=callbacks, 
                      verbose=2)



duration = datetime.now() - start
print("Training completed in time: ", duration)


#%% [markdown]
# Call the accuracy loss plot function to visualize the training performance of the fine tuned model 
# %%
plot_accuracy_loss(history)


#%% [markdown]
# Run the visual test function on the fine tuned model
#%%
test_path = Path(os.path.join(os.getcwd(), root_dir, 'test'))

test_predict_view(test_path, num_examples = 15)

# %%
