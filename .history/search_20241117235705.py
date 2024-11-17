from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class BookSearcher:
    def __init__(self, base_dir: str = "processed_books"):
        self.base_dir = Path(base_dir)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.current_book: Optional[str] = None
        self.current_index = None
        self.current_chunks = None
        
    def load_book(self, book_id: str):
        """특정 책의 인덱스와 청크를 메모리에 로드합니다."""
        if self.current_book == book_id:
            return
            
        book_dir = self.base_dir / book_id
        if not book_dir.exists():
            raise ValueError(f"Book {book_id} not found")
            
        self.current_index = faiss.read_index(str(book_dir / "faiss.index"))
        self.current_chunks = pd.read_csv(book_dir / "chunks.csv", encoding='utf-8')
        self.current_book = book_id
        
    def search(self, query: str, k: int = 10, threshold: float = 0.35) -> List[Dict]:
        """현재 로드된 책에서 쿼리와 관련된 청크를 검색합니다."""
        if not self.current_book:
            raise ValueError("검색하기 전에 책을 선택해주세요.")
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.current_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                chunk = self.current_chunks.iloc[idx]
                results.append({
                    'chunk': chunk['sentence_chunk'],
                    'score': float(score),
                    'chunk_id': int(chunk['chunk_id'])
                })
                
        return results

    def get_current_book(self) -> Optional[str]:
        """현재 선택된 책의 ID를 반환합니다."""
        return self.current_book

    def clear_current_book(self):
        """현재 선택된 책을 초기화합니다."""
        self.current_book = None
        self.current_index = None
        self.current_chunks = None