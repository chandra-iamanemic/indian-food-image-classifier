#%% [markdown]
# Import necessary libraries

#%%
import tensorflow as tf
from tensorflow.keras.layers import  Dense, Flatten, MaxPooling2D 
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
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


#%%
os.getcwd()
image_dir = Path(os.path.join(os.getcwd(), 'images'))
filepath = list(image_dir.glob(r'**/*.jpg'))

labels = []

for x in filepath:
    labels.append(os.path.split(os.path.split(x)[0])[1])

# %%
image_df = pd.DataFrame({"filepath" : filepath, "label" : labels})
image_df["filepath"] = image_df["filepath"].astype(str)


# %%
image_df = image_df.sample(frac=1)


# %%
len(image_df['label'].unique())

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()
# image_df['label'] = label_encoder.fit_transform(image_df['label'])
# image_df['label'].value_counts()
# image_df['label'] = image_df['label'].astype(str)

#%%
train_df, test_df = train_test_split(image_df, test_size = 0.30, shuffle = True, random_state = 1)


# %%
import tensorflow as tf
from tensorflow.keras.layers import  Dense, Flatten, MaxPooling2D 
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2

# %% [markdwon]
# Image Data Generator is used to augment existing images to create variations of the image
# By using this method we can generate a bigger dataset than the one we have by augmenting and creating new images from existing ones

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# %%
train_set = train_datagen.flow_from_dataframe(dataframe = train_df,
                                        preprocessing_function=preprocess_input,
                                           x_col = 'filepath',
                                           y_col = 'label',
                                           target_size = (224,224),
                                           batch_size = 32,
                                           class_mode = 'categorical',
                                           shuffle = True,
                                           seed = 42,
                                           subset = 'training')
validation_set = train_datagen.flow_from_dataframe(dataframe = train_df,
                                           x_col = 'filepath',
                                           y_col = 'label',
                                           target_size = (224,224),
                                           batch_size = 32,
                                           class_mode = 'categorical',
                                           shuffle = True,
                                           seed = 42,
                                           subset = 'validation')
test_set = test_datagen.flow_from_dataframe(dataframe = test_df,
                                           x_col = 'filepath',
                                           y_col = 'label',
                                           target_size = (224,224),
                                           batch_size = 32,
                                           class_mode = 'categorical',
                                           shuffle = False)

print("train set len : ", len(train_set))
print("validation set len : ", len(validation_set))
print("test set len : ", len(test_set))



#%%
sample_batch = train_set.next()

sample_images = sample_batch[0]
sample_labels = sample_batch[1]


for i,j in zip(sample_images, sample_labels):
    # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    cv2.imshow(f"{j}",i)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()


# %%
for i in test_df["filepath"]:
    z = cv2.imread(i)
    cv2.imshow("z",z)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()

# %%
