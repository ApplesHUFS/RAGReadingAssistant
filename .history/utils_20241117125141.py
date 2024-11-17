import numpy as np
from sentence_transformers import SentenceTransformer

MAX_CONTEXT_LENGTH = 5000
RELEVANCE_THRESHOLD = 0.35

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def get_embedding(text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True).astype('float32')