# Use an NVIDIA CUDA base image that matches your CUDA version
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Explicitly install GPU versions of PyTorch and related libraries
RUN pip3 install --upgrade \
    numpy \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    transformers[torch] \
    uvicorn

# Copy and run the download script
COPY download.py .
RUN python3 download.py

# Copy the rest of the application
COPY . .

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]