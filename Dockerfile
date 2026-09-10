FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip

# Install dependencies with explicit error handling
RUN pip install --no-cache-dir -r requirements.txt || (echo "ERROR: Failed to install dependencies" && exit 1)

COPY . .

RUN chmod +x /app/entrypoint.sh

# Verify critical imports work before runtime
RUN python -c "from google import genai; print('✓ Google GenAI imported successfully')"
RUN python -c "import torch; print('✓ PyTorch imported successfully')"
RUN python -c "from loguru import logger; print('✓ Loguru imported successfully')"

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
