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
        """특정 책의 HNSW 인덱스와 청크를 메모리에 로드합니다."""
        if self.current_book == book_id:
            return
            
        book_dir = self.base_dir / book_id
        if not book_dir.exists():
            raise ValueError(f"Book {book_id} not found")
            
        # HNSW 인덱스 로드
        self.current_index = faiss.read_index(str(book_dir / "faiss.index"))
        self.current_chunks = pd.read_csv(book_dir / "chunks.csv", encoding='utf-8')
        self.current_book = book_id
        
        # 검색 품질 설정
        if hasattr(self.current_index, 'hnsw'):
            self.current_index.hnsw.efSearch = 128
    
    def search(self, query: str, k: int = 20, threshold: float = 0.35) -> List[Dict]:
        """HNSW 인덱스를 사용하여 쿼리와 관련된 청크를 검색합니다."""
        if not self.current_book:
            raise ValueError("검색하기 전에 책을 선택해주세요.")
            
        # 쿼리 임베딩 생성 및 정규화
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # 검색 수행
        scores, indices = self.current_index.search(query_embedding, k)
        
        # 결과 필터링 및 반환
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx != -1:
                chunk = self.current_chunks.iloc[idx]
                result = {
                    'chunk': chunk['sentence_chunk'],
                    'score': float(score),
                    'chunk_id': int(chunk['chunk_id'])
                }
                results.append(result)
        
        # 유사도 점수로 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def get_current_book(self) -> Optional[str]:
        """현재 선택된 책의 ID를 반환합니다."""
        return self.current_book

    def clear_current_book(self):
        """현재 선택된 책을 초기화합니다."""
        self.current_book = None
        self.current_index = None
        self.current_chunks = None