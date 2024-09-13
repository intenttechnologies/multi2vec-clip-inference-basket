FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

## Set environment variables
#ENV DEBIAN_FRONTEND=noninteractive
#ENV PATH="/root/.local/bin:$PATH"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV TORCH_CUDNN_SDPA_ENABLED=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Install Poetry
RUN python3.10 -m pip install poetry==1.8.3 setuptools

WORKDIR /app
# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* ./

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Explicitly install GPU versions of PyTorch and related libraries
RUN python3.10 -m pip install --upgrade \
    transformers[torch] \
    uvicorn

# Copy and run the download script
COPY download.py .
RUN python3 download.py

# Copy the rest of the application
COPY . .

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]