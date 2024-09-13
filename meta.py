import json
from transformers import CLIPConfig

class Meta:

  def __init__(self):
    self._config = CLIPConfig.from_pretrained('/app/models/openai_clip').to_dict()
    
    self._config = json.loads(json.dumps(self._config, default=str))

  async def get(self):
    return self._config
