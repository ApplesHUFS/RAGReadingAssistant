from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from readrag.config import settings
from readrag.core.exceptions import SearchError

class BookSearcher:
    def __init__(self):
        self.base_dir = settings.PROCESSED_DIR
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.current_book: Optional[str] = None
        self.current_index = None
        self.current_chunks = None
        
    def load_book(self, book_id: str):
        if self.current_book == book_id:
            return
            
        try:
            book_dir = self.base_dir / book_id
            if not book_dir.exists():
                raise SearchError(f"Book {book_id} not found")
                
            self.current_index = faiss.read_index(str(book_dir / "faiss.index"))
            self.current_chunks = pd.read_csv(book_dir / "chunks.csv", encoding='utf-8')
            self.current_book = book_id
            
            if hasattr(self.current_index, 'hnsw'):
                self.current_index.hnsw.efSearch = settings.HNSW_EF_SEARCH
                
        except Exception as e:
            raise SearchError(f"Error loading book: {e}")
            
    def search(self, query: str) -> List[Dict]:
        if not self.current_book:
            raise SearchError("No book selected for search")
            
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.current_index.search(
                query_embedding,
                settings.SEARCH_K
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= settings.SEARCH_THRESHOLD and idx != -1:
                    chunk = self.current_chunks.iloc[idx]
                    results.append({
                        'chunk': chunk['sentence_chunk'],
                        'score': float(score),
                        'chunk_id': int(chunk['chunk_id'])
                    })
                    
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
            
        except Exception as e:
            raise SearchError(f"Error during search: {e}")
            
    def get_current_book(self) -> Optional[str]:
        return self.current_book
        
    def clear_current_book(self):
        self.current_book = None
        self.current_index = None
        self.current_chunks = None