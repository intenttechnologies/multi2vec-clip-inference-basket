#!/usr/bin/env python3

import logging
import os
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)

clip_model_name = "openai/clip-vit-base-patch32"
base_path = "/app/models"
clip_model_path = os.path.join(base_path, "openai_clip")
clip_processor_path = os.path.join(base_path, "openai_clip_processor")

os.makedirs(base_path, exist_ok=True)

logging.info(f"Downloading OpenAI CLIP model {clip_model_name} from huggingface model hub")

# Download the model files without loading them
model_path = snapshot_download(clip_model_name, local_dir=clip_model_path)

logging.info(f"Model files downloaded to {model_path}")

# Download the processor files
processor = CLIPProcessor.from_pretrained(clip_model_name)
processor.save_pretrained(clip_processor_path)

logging.info(f"Processor files saved to {clip_processor_path}")