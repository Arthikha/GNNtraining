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