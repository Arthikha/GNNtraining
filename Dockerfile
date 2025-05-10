# # Use Python as base image
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Copy dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# RUN mkdir -p models
# COPY models/best_model.pt ./models/best_model.pt

# # Copy training script
# COPY . .

# # Start training
# # CMD ["python", "src/train/train_graphsage.py"]
# CMD ["python", "predict_fraud.py"]


FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/models /app/output
COPY predict_fraud.py .
COPY models/best_model.pt ./models/
RUN chmod -R 777 /app/output
CMD ["python", "predict_fraud.py"]