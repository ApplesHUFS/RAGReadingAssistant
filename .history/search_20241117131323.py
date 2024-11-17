from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from utils import get_embedding, RELEVANCE_THRESHOLD
from thefuzz import fuzz
import re
from konlpy.tag import Okt

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.chunk_data = []
        self.bm25 = None
        self.titles = []
        self.title_to_chunks = {}
        self.okt = Okt()
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        embeddings = []
        self.chunk_data = []
        
        # 정규화된 제목을 저장할 딕셔너리
        normalized_titles = {}
        
        for chunk in chunks:
            embeddings.append(chunk['embedding'])
            self.chunk_data.append(chunk)
            
            title = chunk['pdf_id']
            # 제목 정규화 (한글 특화 처리)
            normalized_title = self._normalize_korean_title(title)
            
            if normalized_title not in normalized_titles:
                self.titles.append(title)
                normalized_titles[normalized_title] = title
                self.title_to_chunks[title] = []
            self.title_to_chunks[title].append(chunk)
        
        # BM25에는 형태소 분석된 토큰을 사용
        tokenized_titles = [self._tokenize_korean_title(title) for title in self.titles]
        self.bm25 = BM25Okapi(tokenized_titles)
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
    
    def _normalize_korean_title(self, title: str) -> str:
        """한글 제목을 정규화하는 헬퍼 함수"""
        # 1. 공백 정규화
        title = ' '.join(title.split())
        # 2. 특수문자 제거 (단, 한글/영문/숫자는 유지)
        title = re.sub(r'[^\w\s가-힣]', '', title)
        # 3. 대소문자 통일
        return title.lower()
    
    def _tokenize_korean_title(self, title: str) -> List[str]:
        """한글 제목을 토큰화하는 헬퍼 함수"""
        # 1. 정규화 먼저 수행
        normalized = self._normalize_korean_title(title)
        # 2. 형태소 분석
        morphs = self.okt.morphs(normalized)
        # 3. 명사 추출
        nouns = self.okt.nouns(normalized)
        # 4. 형태소와 명사를 함께 반환
        return morphs + nouns

    def search_title(self, query: str, threshold: float = 0.15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized_query = self._normalize_korean_title(query)
        tokenized_query = self._tokenize_korean_title(query)
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        matching_results = []
        for idx, title in enumerate(self.titles):
            if normalized_query == self._normalize_korean_title(title):
                matching_results.append({
                    'title': title,
                    'score': 100,
                    'chunks': self.title_to_chunks[title]
                })
                continue

            normalized_title = self._normalize_korean_title(title)

            str_ratio = fuzz.partial_ratio(normalized_query, normalized_title)

            query_chars = self._get_korean_chars(normalized_query)
            title_chars = self._get_korean_chars(normalized_title)
            char_ratio = fuzz.partial_ratio(query_chars, title_chars)
            
            normalized_bm25 = min(100, (bm25_scores[idx] * 50))
            
            final_score = (str_ratio * 0.3) + (char_ratio * 0.3) + (normalized_bm25 * 0.4)
            
            if final_score >= (threshold * 100):
                matching_results.append({
                    'title': title,
                    'score': final_score,
                    'chunks': self.title_to_chunks[title]
                })
   
        matching_results.sort(key=lambda x: x['score'], reverse=True)
        
        similar_titles = [
            {
                'title': result['title'],
                'score': result['score']
            }
            for result in matching_results[:5]
        ]

        if matching_results:
            all_matching_chunks = []
            for result in matching_results[:3]:
                all_matching_chunks.extend(result['chunks'])
            return all_matching_chunks, similar_titles
        
        return [], similar_titles

    def _get_korean_chars(self, text: str) -> str:
        result = []
        for char in text:
            if '가' <= char <= '힣':
                result.append(chr((ord(char) - 0xAC00) // 28 // 21 + 0x1100))
            else:
                result.append(char)
        return ''.join(result)

    def get_all_titles(self) -> List[str]:
        return self.titles

    def search_content(self, query: str, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = get_embedding(query)
        faiss.normalize_L2(query_embedding)
        
        chunk_indices = [i for i, chunk in enumerate(self.chunk_data) if chunk in chunks]
        
        if not chunk_indices:
            return []
            
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
    chunks, similar_titles = faiss_index.search_title(query)
    return chunks

def get_all_titles() -> List[str]:
    return faiss_index.get_all_titles()