#!/usr/bin/env python3

import logging
from transformers import CLIPProcessor, CLIPConfig
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)

clip_model_name = "openai/clip-vit-base-patch32"

logging.info(
    f"Downloading OpenAI CLIP model {clip_model_name} from huggingface model hub"
)

# Download the model files without instantiating the model
model_path = snapshot_download(clip_model_name)

# Save the configuration
config = CLIPConfig.from_pretrained(model_path)
config.save_pretrained('./models/openai_clip')

# Save the processor
processor = CLIPProcessor.from_pretrained(model_path)
processor.save_pretrained('./models/openai_clip_processor')

logging.info("Model and processor saved successfully")