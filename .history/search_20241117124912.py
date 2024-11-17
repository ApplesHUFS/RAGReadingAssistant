from typing import List, Dict, Any
import numpy as np
import faiss
from utils import get_embedding, RELEVANCE_THRESHOLD
from rank_bm25 import BM25Okapi

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.chunk_data = []
        self.bm25 = None
        self.titles = []
        self.title_to_chunks = {}
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        embeddings = []
        self.chunk_data = []
        
        for chunk in chunks:
            embeddings.append(chunk['embedding'])
            self.chunk_data.append(chunk)
            
            title = chunk['pdf_id']
            if title not in self.titles:
                self.titles.append(title)
                self.title_to_chunks[title] = []
            self.title_to_chunks[title].append(chunk)
        
        self.bm25 = BM25Okapi([[word.lower() for word in title.split()] for title in self.titles])
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
    
    def search_title(self, query: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        tokenized_query = [word.lower() for word in query.split()]
        scores = self.bm25.get_scores(tokenized_query)
        
        matching_chunks = []
        for idx, score in enumerate(scores):
            if score > threshold:
                title = self.titles[idx]
                matching_chunks.extend(self.title_to_chunks[title])
        
        return matching_chunks

    def search_content(self, query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = get_embedding(query)
        faiss.normalize_L2(query_embedding)
        
        chunk_indices = [i for i, chunk in enumerate(self.chunk_data) if chunk in chunks]
        
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for score, idx in zip(scores, indices):
            if score >= RELEVANCE_THRESHOLD and idx in chunk_indices:
                results.append({
                    **self.chunk_data[idx],
                    'relevance_score': float(score)
                })
        
        return results

faiss_index = FAISSIndex()

def search(query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    if faiss_index.index is None:
        faiss_index.build_index(chunks)
    
    return faiss_index.search_content(query, chunks, k)

def search_title(query: str) -> List[Dict[str, Any]]:
    return faiss_index.search_title(query)