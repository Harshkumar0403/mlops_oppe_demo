# Use a slim Python base image
FROM python:3.12-slim

# Make Python output unbuffered (good for logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (optional, but often useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the app with uvicorn
# "app:app" -> file `app.py`, variable `app = FastAPI(...)`
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

