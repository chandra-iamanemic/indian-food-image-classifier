#%% [markdown]
# Import necessary libraries
#%%

from datetime import datetime
import os
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings("ignore")
import numpy as np


#%%
root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# Set the name of images directory of our final structured dataset
images_dir = os.path.join(root_dir,'images')
raw_data_path = os.path.join(root_dir,'raw_data')

#%% [markdown]
# Create a structured dataset from raw dataset
#%%
# retrieve existing categories from our dataset
category_list = os.listdir(os.path.join(images_dir, 'train'))

# Append new categories if any is added
new_category_list = os.listdir(os.path.join(root_dir, raw_data_path))

for new_category in new_category_list:
    if new_category not in category_list:
        category_list.append(new_category)
        

#%%
# # Just in case we delete a pre-existing folder for the root directory
# if os.path.exists(os.path.join(os.getcwd(), root_dir)):
#     shutil.rmtree(os.path.join(os.getcwd(), root_dir))

# Create train validation and test folders inside root directory
# Loop through each category in the list
for i in new_category_list:
    
    # For each category create a folder within train, validation and test folders respectively
    if not os.path.exists(f'{images_dir}/train/{i}'):
        os.makedirs(f'{images_dir}/train/{i}')
    if not os.path.exists(f'{images_dir}/validation/{i}'):
        os.makedirs(f'{images_dir}/validation/{i}')
    if not os.path.exists(f'{root_dir}/{images_dir}/test/{i}'):
        os.makedirs(f'{images_dir}/test/{i}')

    source = raw_data_path + '/' + i

    allFileNames = os.listdir(source)

    np.random.shuffle(allFileNames)

    test_split_ratio = 0.3
    test_len = int(len(allFileNames)*test_split_ratio)
    train_FileNames = allFileNames[:-test_len]
    temp_test_FileNames = allFileNames[-test_len:]

    validation_split_ratio = 0.5
    validation_len = int(len(temp_test_FileNames)*validation_split_ratio)

    test_FileNames = temp_test_FileNames[:-validation_len]
    validation_FileNames = temp_test_FileNames[-validation_len:]

    for name in train_FileNames:
        shutil.copy(f'{source}/{name}', f'{images_dir}/train/{i}')

    for name in test_FileNames:
        shutil.copy(f'{source}/{name}', f'{images_dir}/test/{i}')

    for name in validation_FileNames:
        shutil.copy(f'{source}/{name}', f'{images_dir}/validation/{i}')
