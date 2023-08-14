#%%
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

#%%
root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

SAVE_DIR = os.path.join(root_dir, 'raw_data')


# Define the list of food items
food_items = ['sambar', 'rasam', 'curd rice', 'vatha kuzhambu', 'poha', 'biriyani', 'lassi', 'paneer butter masala', 'gobi manchurian']

# Specify the number of images you want to download per food item
num_images_per_item = 300

# Create the save directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

#%%
# Search and download images for each food item
for food_item in food_items:
    query = quote_plus(food_item)  # URL-encoded query

    search_url = f"https://www.google.com/search?q={food_item.replace(' ', '+')}&tbm=isch"

    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Create a subdirectory for each food item
    item_save_dir = os.path.join(SAVE_DIR, food_item)
    if not os.path.exists(item_save_dir):
        os.makedirs(item_save_dir)

    # Download and save images
    image_count = 0
    for img in soup.find_all('img'):
        image_url = img.get('src')
        if image_url and image_url.startswith('http'):
            image_data = requests.get(image_url)
            if image_data.status_code == 200:
                with open(os.path.join(item_save_dir, f'{food_item}_{image_count + 1}.jpg'), 'wb') as f:
                    f.write(image_data.content)
                image_count += 1

                if image_count >= num_images_per_item:
                    break

            print(f'Downloaded: {food_item}_{image_count}.jpg')

print('Image downloading completed.')

#%%

