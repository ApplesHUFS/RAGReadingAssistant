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
        normalized_titles = {}
        
        for chunk in chunks:
            embeddings.append(chunk['embedding'])
            self.chunk_data.append(chunk)
            
            title = chunk['pdf_id']
            normalized_title = self._normalize_korean_title(title)
            
            if normalized_title not in normalized_titles:
                self.titles.append(title)
                normalized_titles[normalized_title] = title
                self.title_to_chunks[title] = []
            self.title_to_chunks[title].append(chunk)
        
        tokenized_titles = [self._tokenize_korean_title(title) for title in self.titles]
        self.bm25 = BM25Okapi(tokenized_titles)
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)

    def search_title(self, query: str, threshold: float = 0.30) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized_query = self._normalize_korean_title(query)
        tokenized_query = self._tokenize_korean_title(query)

        exact_matches = []
        for title in self.titles:
            if normalized_query in self._normalize_korean_title(title):
                exact_matches.append({
                    'title': title,
                    'score': 100,
                    'chunks': self.title_to_chunks[title]
                })
        
        if exact_matches:
            return self._process_matching_results(exact_matches)
        
        matching_results = []
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        for idx, title in enumerate(self.titles):
            normalized_title = self._normalize_korean_title(title)
            
            str_ratio = fuzz.ratio(normalized_query, normalized_title)
            
            query_chars = self._get_korean_chars(normalized_query)
            title_chars = self._get_korean_chars(normalized_title)
            char_ratio = fuzz.ratio(query_chars, title_chars)
            
            normalized_bm25 = min(100, (bm25_scores[idx] * 50))
            
            final_score = (str_ratio * 0.35) + (char_ratio * 0.35) + (normalized_bm25 * 0.3)
            
            if final_score >= (threshold * 100):
                matching_results.append({
                    'title': title,
                    'score': final_score,
                    'chunks': self.title_to_chunks[title]
                })
        
        matching_results.sort(key=lambda x: x['score'], reverse=True)
        return self._process_matching_results(matching_results)

    def _process_matching_results(self, matching_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        similar_titles = [
            {
                'title': result['title'],
                'score': result['score']
            }
            for result in matching_results[:6]
        ]

        if matching_results:
            top_chunks = []
            for result in matching_results[:6]:
                top_chunks.extend(result['chunks'])
            return top_chunks, similar_titles
        
        return [], similar_titles

    def search_content(self, query: str, chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = get_embedding(query)
        faiss.normalize_L2(query_embedding)
        
        chunk_indices = [i for i, chunk in enumerate(self.chunk_data) if chunk in chunks]
        
        if not chunk_indices:
            return []
        
        scores, indices = self.index.search(query_embedding, len(chunk_indices))
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for score, idx in zip(scores, indices):
            if idx in chunk_indices and score >= RELEVANCE_THRESHOLD:
                results.append({
                    **self.chunk_data[idx],
                    'relevance_score': float(score)
                })
            
            if len(results) >= k:
                break
        
        return results

    def _normalize_korean_title(self, title: str) -> str:
        title = ' '.join(title.split())
        title = re.sub(r'[^\w\s가-힣]', '', title)
        return title.lower()
    
    def _tokenize_korean_title(self, title: str) -> List[str]:
        normalized = self._normalize_korean_title(title)
        morphs = self.okt.morphs(normalized)
        nouns = self.okt.nouns(normalized)
        return list(set(morphs + nouns))
    
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
    
faiss_index = FAISSIndex()

def search(query: str, chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    if faiss_index.index is None:
        faiss_index.build_index(chunks)
    
    return faiss_index.search_content(query, chunks, k)

def search_content(query: str, chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    if faiss_index.index is None:
        faiss_index.build_index(chunks)
    
    return faiss_index.search_content(query, chunks, k)

def search_title(query: str) -> List[Dict[str, Any]]:
    chunks, similar_titles = faiss_index.search_title(query)
    print(similar_titles)
    return chunks

def get_all_titles() -> List[str]:
    return faiss_index.get_all_titles()