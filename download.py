#!/usr/bin/env python3

import logging
from huggingface_hub import snapshot_download
import os
import json

logging.basicConfig(level=logging.INFO)

clip_model_name = "openai/clip-vit-base-patch32"

logging.info(f"Downloading OpenAI CLIP model {clip_model_name} from huggingface model hub")

# Download the model files
model_path = snapshot_download(repo_id=clip_model_name)

# Define the paths where we want to save our files
save_path = '/app/models/openai_clip'
processor_path = '/app/models/openai_clip_processor'

# Ensure the directories exist
os.makedirs(save_path, exist_ok=True)
os.makedirs(processor_path, exist_ok=True)

# Copy the necessary files
import shutil

# For the model
shutil.copy(os.path.join(model_path, 'config.json'), save_path)
shutil.copy(os.path.join(model_path, 'pytorch_model.bin'), save_path)

# For the processor
shutil.copy(os.path.join(model_path, 'preprocessor_config.json'), processor_path)

logging.info("Model and processor files saved successfully")