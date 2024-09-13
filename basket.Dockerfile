# Use the existing Docker image as the base
FROM semitechnologies/multi2vec-clip:sentence-transformers-clip-ViT-B-32-multilingual-v1


COPY app.py /app/app.py
COPY clip.py /app/clip.py
