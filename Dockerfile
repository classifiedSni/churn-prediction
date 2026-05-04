FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (LightGBM requires libgomp)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
