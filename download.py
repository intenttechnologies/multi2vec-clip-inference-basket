#!/usr/bin/env python3

import logging
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)

clip_model_name = "openai/clip-vit-base-patch32"

logging.info(
  "Downloading OpenAI CLIP model {} from huggingface model hub".format(clip_model_name)
)

clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_model.save_pretrained('./models/openai_clip')
processor = CLIPProcessor.from_pretrained(clip_model_name)
processor.save_pretrained('./models/openai_clip_processor')

