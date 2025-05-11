# # Use Python as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /app/models /app/output

# Copy training script
COPY predict_fraud.py .

COPY models/best_model.pt ./models/

RUN chmod -R 777 /app/output


CMD ["python", "predict_fraud.py"]


# # Use Python as base image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# RUN mkdir -p /app/models

# # Copy dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip

# # RUN apt-get update && apt-get install -y gcc g++ libblas-dev liblapack-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir -r requirements.txt

# # Copy all files from the current directory to /app
# COPY . /app

# ENV TF_ENABLE_ONEDNN_OPTS=0
# ENV DGLBACKEND=tensorflow

# # Start training
# CMD ["python", "src/train/train_graphsage.py"]
