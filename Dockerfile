FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY farm360_agent/requirements.txt ./requirements.txt

RUN pip install --upgrade pip

# Install dependencies with explicit error handling
RUN pip install --no-cache-dir -r requirements.txt || (echo "ERROR: Failed to install dependencies" && exit 1)

COPY farm360_agent/ .

# Verify critical imports work before runtime
RUN python -c "import google.generativeai; print('✓ Google GenAI imported successfully')"
RUN python -c "import torch; print('✓ PyTorch imported successfully')"
RUN python -c "from loguru import logger; print('✓ Loguru imported successfully')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
