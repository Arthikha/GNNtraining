# Use Python as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY . .

# Start training
CMD ["python", "src/train/train_graphsage.py"]

