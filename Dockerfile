# Use Python as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models
# COPY models/best_model.pt ./models/best_model.pt

# Copy training script
COPY . .

# Start training
CMD ["python", "src/train/train_graphsage.py"]
# CMD ["python", "predict_fraud.py"]

