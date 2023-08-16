#%%
import requests
from bs4 import BeautifulSoup
import os


#%%
root_dir = os.path.join(os.getcwd(), os.pardir)

SAVE_DIR = os.path.join(root_dir, 'raw_data')


# Define the list of food items
food_items = ['sambar', 'rasam', 'curd rice', 'vatha kuzhambu', 'poha', 'biriyani', 'lassi', 'paneer butter masala', 'gobi manchurian']

# Specify the number of images you want to download per food item
num_images_per_item = 300

# Create the save directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


#%%
for food_item in food_items:
    search_url = f"https://www.google.com/search?q={food_item.replace(' ', '+')}&tbm=isch"
    soup = BeautifulSoup(requests.get(search_url).content, "html.parser")

    # Find the "Load more results" button.
    more_results_button = soup.find("div", class_="rg_b_more")

    # Iterate through the pages of search results until we have downloaded enough images.
    while more_results_button and len(image_links) < num_images:
        # Find the image links on the current page.
        image_links = soup.find_all("img")

        # Download the images.
        for image_link in image_links:
        image_url = image_link["src"]
        image_name = image_url.split("/")[-1]
        response = requests.get(image_url)
        with open(os.path.join("images", image_name), "wb") as f:
            f.write(response.content)

        # Get the URL of the next page.
        next_url = more_results_button["href"]

        # Update the search URL.
        search_url = next_url

        # Parse the new search URL.
        soup = BeautifulSoup(requests.get(search_url).content, "html.parser")

    # There are no more pages of search results, or we have downloaded enough images.

#%%

search_url = f"https://www.google.com/search?q={food_item.replace(' ', '+')}&tbm=isch"
response = requests.get(search_url)
soup = BeautifulSoup(response.content, "html.parser")

# Find the image links on the page.
image_links = soup.find_all("img")

# Iterate through the pages of search results until we have found 100 image links.
while len(image_links) < 100:
    # Get the URL of the next page.
    next_url = soup.find("a", class_="pn next")["href"]

    # Update the search URL.
    search_url = next_url

    # Parse the new search URL.
    soup = BeautifulSoup(requests.get(search_url).content, "html.parser")

    # Find the image links on the new page.
    image_links = soup.find_all("img")

