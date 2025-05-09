# Use Python as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory to /app
COPY . /app

RUN chmod +x run_scripts.sh
CMD ["./run_scripts.sh"]

# Start training




# Install system dependencies and Python packages

# RUN apt-get update && apt-get install -y 
# gcc 
# g++ 
# libblas-dev 
# liblapack-dev 
# && pip install --no-cache-dir -r requirements.txt 
# && apt-get clean 
# && rm -rf /var/lib/apt/lists/*

