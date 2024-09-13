import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from contextlib import asynccontextmanager
from clip import Clip, ClipInput, ClipSimilarityInput
from meta import Meta



clip : Clip
meta_config : Meta
logger = getLogger('uvicorn')


@asynccontextmanager
async def lifespan(app: FastAPI):
	global clip
	global meta_config

	cuda_env = os.getenv("ENABLE_CUDA")
	cuda_support=False
	cuda_core=""

	if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
		cuda_support=True
		cuda_core = os.getenv("CUDA_CORE")
		if cuda_core is None or cuda_core == "":
			cuda_core = "cuda:0"
		logger.info(f"CUDA_CORE set to {cuda_core}")
	else:
		logger.info("Running on CPU")

	clip = Clip(cuda_support, cuda_core)
	meta_config = Meta()
	logger.info("Model initialization complete")
	yield

app = FastAPI(lifespan=lifespan)

@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
	response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
async def meta():
	return await meta_config.get()


@app.post("/vectorize")
async def read_item(payload: ClipInput, response: Response):
	try:
		result = await clip.vectorize(payload)
		return {
			"textVectors": result.text_vectors,
			"imageVectors": result.image_vectors
		}
	except Exception as e:
		logger.exception(
			'Something went wrong while vectorizing data.'
		)
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error": str(e)}
	

@app.post("/similarity")
async def similarity(payload: ClipSimilarityInput, response: Response):
	try:
		result = await clip.vectorize(payload)
		return {
			"scores": result.scores,
		}
	except Exception as e:
		logger.exception(
			'Something went wrong while vectorizing data.'
		)
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error": str(e)}
