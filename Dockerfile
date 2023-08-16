# Use an appropriate base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Specify the command to run the training script
CMD ["python", "src/train.py"]
