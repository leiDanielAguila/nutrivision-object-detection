FROM python:3.9-slim

WORKDIR /app

# Install required system dependencies for PIL and OpenCV (used by YOLO)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model
COPY . .

# Make sure the model directory exists
RUN mkdir -p app

# Create an empty file as a placeholder if the model doesn't exist locally
# (you'll need to upload your model to Render or fetch it during startup)
RUN touch app/nutrivision_v3.pt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]