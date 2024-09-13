import io
import base64
from os import path
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List
import os


# _parse_image decodes the base64 and parses the image bytes into a
# PIL.Image. If the image is not in RGB mode, e.g. for PNGs using a palette,
# it will be converted to RGB. This makes sure that they work with
# SentenceTransformers/Huggingface Transformers which seems to require a (3,
# height, width) tensor
def parse_image(base64_encoded_image_string):
    image_bytes = base64.b64decode(base64_encoded_image_string)
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


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


class ClipInferenceOpenAI:
    clip_model: CLIPModel
    processor: CLIPProcessor
    lock: Lock

    def __init__(self, cuda, cuda_core):
        self.lock = Lock()
        self.device = 'cpu'
        if cuda:
            self.device = cuda_core

        base_path = "/app/models"
        clip_model_path = os.path.join(base_path, "openai_clip")
        clip_processor_path = os.path.join(base_path, "openai_clip_processor")

        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_processor_path)

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
                image_files = [parse_image(image) for image in payload.images]
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

    def similarity(self, payload: ClipSimilarityInput) -> ClipSimilarityResult:
        try:
            self.lock.acquire()
            image = parse_image(payload.image)

            image = Image.open(str(image))
            texts = payload.texts
            # text = [t.strip() for t in text.split("|")]
            inputs = self.processor(
                text=texts, images=image, return_tensors="pt", padding=True
            ).to(self.device)

            # self.clip_model.

            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            result = ClipSimilarityResult(scores=probs.tolist()[0])
            return result
        finally:
            self.lock.release()


class Clip:
    clip: ClipInferenceOpenAI
    executor: ThreadPoolExecutor

    def __init__(self, cuda, cuda_core):
        self.executor = ThreadPoolExecutor()

        self.clip = ClipInferenceOpenAI(cuda, cuda_core)

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

