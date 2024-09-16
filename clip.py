import io
import base64
from os import path
from abc import ABC, abstractmethod
from typing import Union
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
import open_clip
import torch
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List


class ClipInput(BaseModel):
	texts: list = []
	images: list = []


class ClipResult:
	text_vectors: list = []
	image_vectors: list = []

	def __init__(self, text_vectors, image_vectors):
		self.text_vectors = text_vectors
		self.image_vectors = image_vectors

class ClipSimilarityInput(BaseModel):
    image: str
    texts: List[str]
    
class ClipSimilarityResult:
    scores: List[float]

    def __init__(self, scores: List[float]):
        self.scores = scores

class ClipInferenceABS(ABC):
	"""
	Abstract class for Clip Inference models that should be inherited from.
	"""

	@abstractmethod
	def vectorize(self, payload: ClipInput) -> ClipResult:
		...

	@abstractmethod
	def similarity(self, payload: ClipSimilarityInput) -> ClipSimilarityResult:
		...

class ClipInferenceSentenceTransformers(ClipInferenceABS):
	img_model: SentenceTransformer
	text_model: SentenceTransformer
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		device = 'cpu'
		if cuda:
			device = cuda_core

		self.img_model = SentenceTransformer('./models/clip', device=device)
		self.text_model = SentenceTransformer('./models/text', device=device)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				text_vectors = (
					self.text_model
					.encode(payload.texts, convert_to_tensor=True)
					.tolist()
				)
			finally:
				self.lock.release()
		
		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [_parse_image(image) for image in payload.images]
				image_vectors = (
					self.img_model
					.encode(image_files, convert_to_tensor=True)
					.tolist()
				)
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)
  
	def similarity(self, payload: ClipSimilarityInput) -> ClipSimilarityResult:
		try:
			self.lock.acquire()
			image = _parse_image(payload.image)
			text = payload.texts

			# Convert the image to an embedding
			image_embedding = self.img_model.encode(image, convert_to_tensor=True)

			# Convert the text to an embedding
			text_embedding = self.text_model.encode(text, convert_to_tensor=True)

			# Compute cosine similarity
			similarity = util.cos_sim(image_embedding, text_embedding)

			# Without softmax the results are way too close together
			# scores = similarity[0]		
   
			# Apply softmax to amplify differences
			scores = torch.nn.functional.softmax(similarity[0] / 0.1, dim=0)

			# Convert similarity tensor to a list of floats
			scores = scores.tolist()

			return ClipSimilarityResult(scores=scores)
		finally:
			self.lock.release()


class ClipInferenceOpenAI:
	clip_model: CLIPModel
	processor: CLIPProcessor
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core
		self.clip_model = CLIPModel.from_pretrained('./models/openai_clip').to(self.device)
		self.processor = CLIPProcessor.from_pretrained('./models/openai_clip_processor')

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				inputs = self.processor(
					text=payload.texts,
					return_tensors="pt",
					padding=True,
				).to(self.device)

				# Taken from the HuggingFace source code of the CLIPModel
				text_outputs = self.clip_model.text_model(**inputs)
				text_embeds = text_outputs[1]
				text_embeds = self.clip_model.text_projection(text_embeds)

				# normalized features
				text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
				text_vectors = text_embeds.tolist()
			finally:
				self.lock.release()


		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [_parse_image(image) for image in payload.images]
				inputs = self.processor(
					images=image_files,
					return_tensors="pt",
					padding=True,
				).to(self.device)

				# Taken from the HuggingFace source code of the CLIPModel
				vision_outputs = self.clip_model.vision_model(**inputs)
				image_embeds = vision_outputs[1]
				image_embeds = self.clip_model.visual_projection(image_embeds)

				# normalized features
				image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
				image_vectors = image_embeds.tolist()
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class ClipInferenceOpenCLIP:
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core

		cache_dir = './models/openclip'
		with open(path.join(cache_dir, "config.json")) as user_file:
			config = json.load(user_file)

		model_name = config['model_name']
		pretrained = config['pretrained']

		model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir, device=self.device)
		if cuda:
			model = model.to(device=self.device)

		self.clip_model = model
		self.preprocess = preprocess
		self.tokenizer = open_clip.get_tokenizer(model_name)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				with torch.no_grad(), torch.cuda.amp.autocast():
					text = self.tokenizer(payload.texts).to(self.device)
					text_features = self.clip_model.encode_text(text).to(self.device)
					text_features /= text_features.norm(dim=-1, keepdim=True)
				text_vectors = text_features.tolist()
			finally:
				self.lock.release()

		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [self.preprocess_image(image) for image in payload.images]
				image_vectors = [self.vectorize_image(image) for image in image_files]
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)

	def preprocess_image(self, base64_encoded_image_string):
		image_bytes = base64.b64decode(base64_encoded_image_string)
		img = Image.open(io.BytesIO(image_bytes))
		return self.preprocess(img).unsqueeze(0).to(device=self.device)

	def vectorize_image(self, image):
		with torch.no_grad(), torch.cuda.amp.autocast():
			image_features = self.clip_model.encode_image(image).to(self.device)
			image_features /= image_features.norm(dim=-1, keepdim=True)

		return image_features.tolist()[0]


class Clip:

	clip: Union[ClipInferenceOpenAI, ClipInferenceSentenceTransformers, ClipInferenceOpenCLIP]
	executor: ThreadPoolExecutor

	def __init__(self, cuda, cuda_core):
		self.executor = ThreadPoolExecutor()

		if path.exists('./models/openai_clip'):
			self.clip = ClipInferenceOpenAI(cuda, cuda_core)
		elif path.exists('./models/openclip'):
			self.clip = ClipInferenceOpenCLIP(cuda, cuda_core)
		else:
			self.clip = ClipInferenceSentenceTransformers(cuda, cuda_core)

	async def vectorize(self, payload: ClipInput):
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		return await asyncio.wrap_future(self.executor.submit(self.clip.vectorize, payload))

	async def similarity(self, payload: ClipSimilarityInput):
		return await asyncio.wrap_future(self.executor.submit(self.clip.similarity, payload))


# _parse_image decodes the base64 and parses the image bytes into a
# PIL.Image. If the image is not in RGB mode, e.g. for PNGs using a palette,
# it will be converted to RGB. This makes sure that they work with
# SentenceTransformers/Huggingface Transformers which seems to require a (3,
# height, width) tensor
def _parse_image(base64_encoded_image_string):
	image_bytes = base64.b64decode(base64_encoded_image_string)
	img = Image.open(io.BytesIO(image_bytes))

	if img.mode != 'RGB':
		img = img.convert('RGB')
	return img
