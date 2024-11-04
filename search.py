from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from utils import cosine_similarity, get_embedding, RELEVANCE_THRESHOLD

def search(query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = get_embedding(query)

    scored_results = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk['embedding'])
        
        if score >= RELEVANCE_THRESHOLD:
            scored_results.append({
                **chunk,
                'relevance_score': score
            })

    scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_results[:k]