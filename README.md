# indian-food-image-classifier
Classifier to identify Indian food based on input image

# Docker Setup


docker build -t ml-training-image .

docker run -v "$(pwd)/exports:/app/exports" ml-training-image

docker run --cpus=0.0 --memory=0 -v "$(pwd)/exports:/app/exports" ml-training-image


 ## Video snippets of the project
 You can look at some visuals on these links :
 
  * [Live Predictions Demo 1](https://drive.google.com/file/d/1zV1f845e7bwsS113sX7Cj9B1bc_N21GX/view?usp=drive_link)
  * [Live Predictions Demo 2](https://drive.google.com/file/d/1KhxbvOvG-6Obuxw2pOX8F8kTzfFraFZ_/view?usp=drive_link) 
  * [Live Predictions Demo 3](https://drive.google.com/file/d/1WryvOGwt72YQPG_LuwGobtYnGRZEabUB/view?usp=drive_link)
