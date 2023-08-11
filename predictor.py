#%%
import keras
import cv2
import os
import pandas as pd
from pathlib import Path
import numpy as np

model_path = "mobilenetv2.h5"
model = keras.models.load_model(model_path)

print(model.summary())

#%%
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def process_input_image(path):
    img = load_img(path, target_size= (224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255
    return img_array


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


#%%
category_list = os.listdir(os.path.join(os.getcwd(), 'images', 'train'))

# Set the name of root directory of our final structured dataset
root_dir = 'images'


#%%
# Open a text file in write mode
with open('labels_list.txt', 'w') as file:
    for item in category_list:
        file.write(str(item) + '\n')


#%%
print(model.summary())

# %%
#Setting dataset path for train and test sets 
train_path = f"{root_dir}/train"
validation_path = f"{root_dir}/validation"
test_path = f"{root_dir}/test"

test_path = Path(os.path.join(os.getcwd(), root_dir, 'test'))

test_predict_view(test_path, num_examples = 15)

#%%
