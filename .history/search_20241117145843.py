from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from utils import get_embedding
from thefuzz import fuzz
import re
from konlpy.tag import Komoran
import pickle
import os
from collections import defaultdict

class SearchResult:
    def __init__(self, chunks: List[Dict[str, Any]], similar_titles: List[Dict[str, str]]):
        self.chunks = chunks
        self.similar_titles = similar_titles

    def __bool__(self):
        return bool(self.chunks)

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.chunk_data = []
        self.bm25 = None
        self.titles = []
        self.title_to_chunks = {}
        self.komoran = Komoran()
        self.chunk_to_context = defaultdict(list)
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        """인덱스 초기 구축"""
        embeddings = []
        self.chunk_data = []
        self.title_to_chunks = {}
        all_texts = []
        
        # 1단계: 청크 데이터 초기 구성
        for chunk in chunks:
            title = chunk['pdf_id']
            if title not in self.title_to_chunks:
                self.titles.append(title)
                self.title_to_chunks[title] = []
            self.title_to_chunks[title].append(chunk)
        
        # 2단계: 문맥 정보 구축 및 임베딩 준비
        for i, chunk in enumerate(chunks):
            # 문맥 정보 저장
            context_start = max(0, i - 2)
            context_end = min(len(chunks), i + 3)
            self.chunk_to_context[chunk['sentence_chunk']] = chunks[context_start:context_end]
            
            # 임베딩 및 텍스트 준비
            embeddings.append(chunk['embedding'])
            self.chunk_data.append(chunk)
            
            # BM25를 위한 텍스트 준비
            text = f"{chunk['pdf_id']} {chunk['sentence_chunk']}"
            tokens = self._tokenize_text(text)
            all_texts.append(tokens)
        
        # 3단계: FAISS 인덱스 구축
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        # 4단계: BM25 인덱스 구축
        self.bm25 = BM25Okapi(all_texts)

    def search_content(self, query: str, chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """컨텐츠 검색 - 임베딩과 BM25 결합"""
        if not chunks:
            return []
            
        # 1. 임베딩 기반 검색
        query_embedding = get_embedding(query)
        faiss.normalize_L2(query_embedding)
        
        valid_indices = [i for i, chunk in enumerate(self.chunk_data) if chunk in chunks]
        if not valid_indices:
            return []
        
        # 전체 검색 후 필터링
        n_search = min(len(valid_indices) * 2, len(self.chunk_data))
        scores, indices = self.index.search(query_embedding, n_search)
        scores, indices = scores[0], indices[0]
        
        # 2. BM25 검색
        query_tokens = self._tokenize_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 3. 결과 결합 및 정렬
        results = []
        seen_contents = set()
        
        # 스코어 정규화 및 결합
        max_embed_score = max(scores) if scores.size > 0 else 1
        max_bm25_score = max(bm25_scores) if len(bm25_scores) > 0 else 1
        
        scored_chunks = []
        for i, (embed_score, idx) in enumerate(zip(scores, indices)):
            if idx not in valid_indices:
                continue
                
            chunk = self.chunk_data[idx]
            content = chunk['sentence_chunk']
            
            if content in seen_contents:
                continue
                
            # 정규화된 점수 결합
            norm_embed_score = embed_score / max_embed_score
            norm_bm25_score = bm25_scores[idx] / max_bm25_score
            
            # 가중치 적용
            combined_score = (0.7 * norm_embed_score) + (0.3 * norm_bm25_score)
            
            if combined_score >= 0.2:  # 낮은 임계값 적용
                scored_chunks.append((combined_score, chunk))
                seen_contents.add(content)
        
        # 점수순 정렬
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # 4. 문맥을 고려한 결과 구성
        final_results = []
        seen_contents = set()
        
        for score, chunk in scored_chunks[:k]:
            if len(final_results) >= k * 2:  # 문맥 포함하여 더 많은 결과 허용
                break
                
            content = chunk['sentence_chunk']
            if content in seen_contents:
                continue
                
            # 문맥 청크 포함
            context_chunks = self.chunk_to_context[content]
            for ctx_chunk in context_chunks:
                if ctx_chunk['sentence_chunk'] not in seen_contents:
                    final_results.append({
                        **ctx_chunk,
                        'relevance_score': float(score)
                    })
                    seen_contents.add(ctx_chunk['sentence_chunk'])
        
        return final_results[:k]  # 최종 k개로 제한

    def search_title(self, query: str, threshold: float = 0.2) -> SearchResult:
        """제목 검색 - 퍼지 매칭과 토큰 기반 검색 결합"""
        normalized_query = self._normalize_korean_title(query)
        tokenized_query = self._tokenize_text(normalized_query)
        
        # 1. 정확한 매칭 확인
        exact_matches = []
        for title in self.titles:
            normalized_title = self._normalize_korean_title(title)
            
            # 저자 정보 제외한 매칭
            title_only = self._remove_author(normalized_title)
            query_only = self._remove_author(normalized_query)
            
            if query_only in title_only or title_only in query_only:
                exact_matches.append({
                    'title': title,
                    'score': 100,
                    'chunks': self.title_to_chunks[title]
                })
        
        if exact_matches:
            return self._process_matching_results(exact_matches)
        
        # 2. 유사도 기반 매칭
        matching_results = []
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        for idx, title in enumerate(self.titles):
            normalized_title = self._normalize_korean_title(title)
            title_only = self._remove_author(normalized_title)
            query_only = self._remove_author(normalized_query)
            
            # 다양한 유사도 측정
            str_ratio = fuzz.ratio(query_only, title_only)
            char_ratio = fuzz.ratio(
                self._get_korean_chars(query_only),
                self._get_korean_chars(title_only)
            )
            token_ratio = fuzz.token_sort_ratio(query_only, title_only)
            
            # BM25 점수 정규화
            normalized_bm25 = min(100, (bm25_scores[idx] * 60))
            
            # 최종 점수 계산
            final_score = (
                str_ratio * 0.35 +
                char_ratio * 0.35 +
                token_ratio * 0.2 +
                normalized_bm25 * 0.1
            )
            
            if final_score >= (threshold * 100):
                matching_results.append({
                    'title': title,
                    'score': final_score,
                    'chunks': self.title_to_chunks[title]
                })
        
        matching_results.sort(key=lambda x: x['score'], reverse=True)
        return self._process_matching_results(matching_results)

    def _process_matching_results(self, matching_results: List[Dict[str, Any]]) -> SearchResult:
        """검색 결과 처리"""
        if not matching_results:
            return SearchResult([], [])
            
        similar_titles = [
            {
                'title': result['title'],
                'score': result['score']
            }
            for result in matching_results[:8]
        ]
        
        all_chunks = []
        for result in matching_results[:6]:  # 상위 6개 제목까지만 포함
            all_chunks.extend(result['chunks'])
            
        return SearchResult(all_chunks, similar_titles)

    def _normalize_korean_title(self, title: str) -> str:
        """한글 제목 정규화"""
        title = ''.join(title.split())
        title = re.sub(r'\([^)]*\)', '', title)  # 괄호 내용 제거
        title = re.sub(r'[^\w\s가-힣]', '', title)  # 특수문자 제거
        return title.lower()

    def _tokenize_text(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        # 기본 형태소 분석
        morphs = self.komoran.morphs(text)
        # 명사 추출
        nouns = self.komoran.nouns(text)
        # 중복 제거 및 결합
        return list(set(morphs + nouns))

    def _remove_author(self, title: str) -> str:
        """저자 정보 제거"""
        parts = title.split()
        if len(parts) > 1 and len(parts[-1]) <= 3:
            return ''.join(parts[:-1])
        return title
    
    def _get_korean_chars(self, text: str) -> str:
        """한글 자모 추출"""
        result = []
        for char in text:
            if '가' <= char <= '힣':
                result.append(chr((ord(char) - 0xAC00) // 28 // 21 + 0x1100))
            else:
                result.append(char)
        return ''.join(result)

    def save_index(self, directory: str):
        """인덱스 저장"""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        other_data = {
            'chunk_data': self.chunk_data,
            'titles': self.titles,
            'title_to_chunks': self.title_to_chunks,
            'bm25': self.bm25,
            'chunk_to_context': dict(self.chunk_to_context)
        }
        
        with open(os.path.join(directory, "index_data.pkl"), 'wb') as f:
            pickle.dump(other_data, f)
    
    def load_index(self, directory: str):
        """인덱스 로드"""
        self.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        with open(os.path.join(directory, "index_data.pkl"), 'rb') as f:
            other_data = pickle.load(f)
            
        self.chunk_data = other_data['chunk_data']
        self.titles = other_data['titles']
        self.title_to_chunks = other_data['title_to_chunks']
        self.bm25 = other_data['bm25']
        self.chunk_to_context = defaultdict(list, other_data['chunk_to_context'])

    def get_all_titles(self) -> List[str]:
        """모든 제목 반환"""
        return self.titles

faiss_index = FAISSIndex()

def save_index(directory: str = "saved_index"):
    faiss_index.save_index(directory)

def load_index(directory: str = "saved_index"):
    faiss_index.load_index(directory)

def search(query: str, chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    if faiss_index.index is None:
        faiss_index.build_index(chunks)
    return faiss_index.search_content(query, chunks, k)

def search_title(query: str) -> List[Dict[str, Any]]:
    result = faiss_index.search_title(query)
    if result.similar_titles:
        print(result.similar_titles)
    return result.chunks

def get_all_titles() -> List[str]:
    return faiss_index.get_all_titles()