import json
from os import path
from transformers import CLIPConfig
from transformers import AutoConfig

class Meta:

  def __init__(self):
    self._config = CLIPConfig.from_pretrained('./models/openai_clip').to_dict()
    
    self._config = json.loads(json.dumps(self._config, default=str))

  async def get(self):
    return self._config
