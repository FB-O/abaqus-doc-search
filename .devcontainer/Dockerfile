
# Use a Python base image (explicit version for stability)
FROM python:3.13-slim-bookworm

# Set environment variables (avoid Python bytecode, ensure UTF-8)
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Set working directory
WORKDIR /workspace

# Copy dependency file first (optimizes caching)
COPY requirements.txt .

# Update package lists
RUN apt-get update 

# Install system dependencies
RUN apt-get install --no-install-recommends -y \
    git \
    curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Ugrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
