from typing import List, Dict, Any
import numpy as np
import faiss
from utils import get_embedding, RELEVANCE_THRESHOLD, model 

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.chunk_data = []
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        embeddings = []
        self.chunk_data = []
        
        for chunk in chunks:
            embeddings.append(chunk['embedding'])
            self.chunk_data.append({
                'pdf_id': chunk['pdf_id'],
                'page_number': chunk['page_number'],
                'sentence_chunk': chunk['sentence_chunk']
            })
        
        embeddings_array = np.array(embeddings).astype('float32')
        print(f"Index embedding shape: {embeddings_array.shape}")  # 디버깅용
        
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)  
        self.index.add(embeddings_array)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # 동일한 모델을 사용하여 쿼리 임베딩 생성
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        print(f"Query embedding shape: {query_embedding.shape}")  # 디버깅용
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]  
        indices = indices[0]  
        
        results = []
        for score, idx in zip(scores, indices):
            if score >= RELEVANCE_THRESHOLD:
                chunk_info = self.chunk_data[idx]
                results.append({
                    **chunk_info,
                    'relevance_score': float(score)
                })
        
        return results

faiss_index = FAISSIndex()

def search(query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    if faiss_index.index is None:
        faiss_index.build_index(chunks)
    
    return faiss_index.search(query, k)