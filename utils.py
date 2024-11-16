import numpy as np
from sentence_transformers import SentenceTransformer
import functools
from typing import List

MAX_CONTEXT_LENGTH = 5000

RELEVANCE_THRESHOLD = 0.35

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def get_embedding(text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True)[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))