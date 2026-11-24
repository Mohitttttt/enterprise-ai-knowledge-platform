FROM python:3.14-slim

# Prevent Python from creating unnecessary cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set application directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN python -m pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files
COPY requirements.txt .

# Install application dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Create persistent storage directories
RUN mkdir -p /app/data/chroma /app/uploads

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]