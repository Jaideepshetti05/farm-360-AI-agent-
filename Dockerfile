FROM python:3.11-slim

# Install system dependencies required for OpenCV and Machine Learning
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements and install dependencies
# We assume the parent directory is copied so farm360_agent can access ../models
COPY farm360_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire ml models environment (Agent, Models, Data)
COPY . /app

# Set Working directory directly to backend
WORKDIR /app/farm360_agent

# Expose FastAPI Port
EXPOSE 8000

# Run FastAPI Server leveraging Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
