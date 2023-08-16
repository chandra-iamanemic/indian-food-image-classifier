#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

from datetime import datetime
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

#%%
gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#%%
# From the raw path we can make a list of all the categories or classes by listing the folder names
# root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
root_dir = "/app"
print("Root Dir : ", root_dir)
images_dir = os.path.join(root_dir,'images')
print("Images Dir : ", images_dir)


# %%
#Setting dataset path for train and test sets 
train_path = f"{images_dir}/train"
validation_path = f"{images_dir}/validation"
test_path = f"{images_dir}/test"

category_list = os.listdir(train_path)

#%%

# Image Data Generator is used to augment existing images to create variations of the image
# By using this method we can generate a bigger dataset than the one we have by augmenting and creating new images from existing ones

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




# Import TF Requirements
#%%
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

IMAGE_SIZE = [224, 224, 3]
base_model = MobileNetV2(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)

base_model.trainable = False
base_model.summary()

# %%
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D

global_average_layer = GlobalAveragePooling2D()



#%%
num_classes = len(category_list)
prediction_layer = Dense(num_classes, activation='softmax')

#%%


model = Sequential([
   base_model,
   global_average_layer,
   prediction_layer
])

model.summary()

#%%
# We initialize our parameters for training such as the optimizer, loss and learning rates

adam = optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# %%
# Generate a unique timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the filename with the timestamp
model_filename = f'mobilenetv2_{timestamp}.h5'

#%%
checkpoint = ModelCheckpoint(filepath=os.path.join(root_dir, 'exports', model_filename), 
                               verbose=2, 
                               save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()

history = model.fit(train_set,
                      validation_data=validation_set,
                      epochs=25,
                      validation_steps=len(validation_set),
                      callbacks=callbacks, 
                      verbose=2)


duration = datetime.now() - start
print("Training completed in time: ", duration)








# %%
### FINE TUNING ###
## UNFREEZING MOBILENET V2 BASE LAYERS ###

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

checkpoint = ModelCheckpoint(filepath=os.path.join(root_dir, 'exports', model_filename), 
                               verbose=2, save_best_only=True)
callbacks = [checkpoint]
start = datetime.now()

history_finetuned = model.fit(train_set,
                      validation_data=validation_set,
                      epochs=25,
                      validation_steps=len(validation_set),
                      callbacks=callbacks, 
                      verbose=2)



duration = datetime.now() - start
print("Training completed in time: ", duration)


#%%
# Open a text file in write mode
with open(os.path.join(root_dir, 'exports', 'labels_list.txt'), 'w') as file:
    for item in category_list:
        file.write(str(item) + '\n')
