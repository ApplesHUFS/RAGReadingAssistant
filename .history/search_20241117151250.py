from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from utils import get_embedding
from thefuzz import fuzz
import re
import pickle
import os
from collections import defaultdict

class SimpleKoreanTokenizer:
    def __init__(self):
        self.cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
        self.jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
        self.jong = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

    def morphs(self, text: str) -> List[str]:
        words = text.split()
        tokens = []
        for word in words:
            if len(word) > 1:
                tokens.extend(self._split_syllables(word))
            tokens.append(word)
        return list(set(tokens))
        
    def nouns(self, text: str) -> List[str]:
        return [word for word in text.split() if len(word) >= 2]

    def _split_syllables(self, text: str) -> List[str]:
        result = []
        for char in text:
            if '가' <= char <= '힣':
                code = ord(char) - 0xAC00
                jong = code % 28
                jung = (code // 28) % 21
                cho = code // 28 // 21
                
                result.append(self.cho[cho])
                if jung < len(self.jung):
                    result.append(self.jung[jung])
                if jong > 0 and jong < len(self.jong):
                    result.append(self.jong[jong-1])
        return result

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
        self.tokenizer = SimpleKoreanTokenizer()
        self.chunk_to_context = defaultdict(list)
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        embeddings = []
        self.chunk_data = []
        self.title_to_chunks = {}
        all_texts = []
        
        for chunk in chunks:
            title = chunk['pdf_id']
            if title not in self.title_to_chunks:
                self.titles.append(title)
                self.title_to_chunks[title] = []
            self.title_to_chunks[title].append(chunk)
        
        for i, chunk in enumerate(chunks):
            context_start = max(0, i - 2)
            context_end = min(len(chunks), i + 3)
            self.chunk_to_context[chunk['sentence_chunk']] = chunks[context_start:context_end]
            
            embeddings.append(chunk['embedding'])
            self.chunk_data.append(chunk)
            
            text = f"{chunk['pdf_id']} {chunk['sentence_chunk']}"
            tokens = self._tokenize_text(text)
            all_texts.append(tokens)
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        self.bm25 = BM25Okapi(all_texts)

    def search_content(self, query: str, chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        query_embedding = get_embedding(query)
        faiss.normalize_L2(query_embedding)
        
        valid_indices = [i for i, chunk in enumerate(self.chunk_data) if chunk in chunks]
        if not valid_indices:
            return []
        
        n_search = min(len(valid_indices) * 2, len(self.chunk_data))
        scores, indices = self.index.search(query_embedding, n_search)
        scores, indices = scores[0], indices[0]
        
        query_tokens = self._tokenize_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        results = []
        seen_contents = set()
        
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
                
            norm_embed_score = embed_score / max_embed_score
            norm_bm25_score = bm25_scores[idx] / max_bm25_score
            
            combined_score = (0.7 * norm_embed_score) + (0.3 * norm_bm25_score)
            
            if combined_score >= 0.2:
                scored_chunks.append((combined_score, chunk))
                seen_contents.add(content)
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        final_results = []
        seen_contents = set()
        
        for score, chunk in scored_chunks[:k]:
            if len(final_results) >= k * 2:
                break
                
            content = chunk['sentence_chunk']
            if content in seen_contents:
                continue
                
            context_chunks = self.chunk_to_context[content]
            for ctx_chunk in context_chunks:
                if ctx_chunk['sentence_chunk'] not in seen_contents:
                    final_results.append({
                        **ctx_chunk,
                        'relevance_score': float(score)
                    })
                    seen_contents.add(ctx_chunk['sentence_chunk'])
        
        return final_results[:k]

    def search_title(self, query: str, threshold: float = 0.15) -> SearchResult:
        normalized_query = self._normalize_korean_title(query)
        query_parts = normalized_query.split()
        
        exact_matches = []
        partial_matches = []
        
        for title in self.titles:
            normalized_title = self._normalize_korean_title(title)
            title_parts = normalized_title.split()
            
            title_match_score = 0
            author_match_score = 0
            
            if len(query_parts) > 1:
                query_author = query_parts[-1]
                if any(part == query_author for part in title_parts):
                    author_match_score = 100
            
            query_title = query_parts[0] if len(query_parts) > 1 else normalized_query
            title_only = self._remove_author(normalized_title)
            
            if query_title == title_only:
                title_match_score = 100
            elif query_title in title_only or title_only in query_title:
                match_length = len(query_title if query_title in title_only else title_only)
                total_length = max(len(query_title), len(title_only))
                title_match_score = (match_length / total_length) * 100
            else:
                title_chars = self._get_korean_chars(title_only)
                query_chars = self._get_korean_chars(query_title)
                chosung_ratio = fuzz.ratio(query_chars, title_chars)
                
                title_decomposed = self._decompose_hangul(title_only)
                query_decomposed = self._decompose_hangul(query_title)
                jamo_ratio = fuzz.ratio(query_decomposed, title_decomposed)
                
                title_tokens = set(self._tokenize_text(title_only))
                query_tokens = set(self._tokenize_text(query_title))
                common_tokens = title_tokens & query_tokens
                token_ratio = len(common_tokens) * 100 / max(len(query_tokens), 1)
                
                edit_ratio = fuzz.ratio(query_title, title_only)
                token_sort_ratio = fuzz.token_sort_ratio(query_title, title_only)
                
                title_match_score = (
                    chosung_ratio * 0.3 +
                    jamo_ratio * 0.2 +
                    token_ratio * 0.2 +
                    edit_ratio * 0.15 +
                    token_sort_ratio * 0.15
                )
            
            if len(query_parts) > 1:
                final_score = (title_match_score * 0.6) + (author_match_score * 0.4)
            else:
                final_score = title_match_score
            
            if final_score >= 80:
                exact_matches.append({
                    'title': title,
                    'score': final_score,
                    'chunks': self.title_to_chunks[title]
                })
            elif final_score >= (threshold * 100):
                partial_matches.append({
                    'title': title,
                    'score': final_score,
                    'chunks': self.title_to_chunks[title]
                })
        
        all_matches = exact_matches + partial_matches
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        return self._process_matching_results(all_matches)

    def _decompose_hangul(self, text: str) -> str:
        result = []
        for char in text:
            if '가' <= char <= '힣':
                code = ord(char) - 0xAC00
                jong = code % 28
                jung = (code // 28) % 21
                cho = code // 28 // 21
                
                result.append(chr(0x1100 + cho))
                result.append(chr(0x1161 + jung))
                if jong > 0:
                    result.append(chr(0x11A7 + jong))
            else:
                result.append(char)
        return ''.join(result)

    def _normalize_korean_title(self, title: str) -> str:
        title = re.sub(r'\s+', ' ', title.strip())
        title = re.sub(r'\([^)]*\)', '', title)
        title = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', title)
        
        number_map = {
            '일': '1', '이': '2', '삼': '3', '사': '4', '오': '5',
            '육': '6', '칠': '7', '팔': '8', '구': '9'
        }
        for kor, num in number_map.items():
            title = title.replace(kor, num)
        
        return title.lower()

    def _tokenize_text(self, text: str) -> List[str]:
        normalized = self._normalize_korean_title(text)
        morphs = self.tokenizer.morphs(normalized)
        nouns = self.tokenizer.nouns(normalized)
        return list(set(morphs + nouns))

    def _remove_author(self, title: str) -> str:
        parts = title.split()
        if len(parts) <= 1:
            return title
            
        last_part = parts[-1]
        if len(last_part) <= 3 and all('가' <= c <= '힣' for c in last_part):
            return ' '.join(parts[:-1])
        
        if len(parts) >= 2 and parts[-2].endswith('의'):
            return ' '.join(parts[:-2])
            
        return title

    def _get_korean_chars(self, text: str) -> str:
        result = []
        for char in text:
            if '가' <= char <= '힣':
                result.append(chr((ord(char) - 0xAC00) // 28 // 21 + 0x1100))
            else:
                result.append(char)
        return ''.join(result)

    def _process_matching_results(self, matching_results: List[Dict[str, Any]]) -> SearchResult:
        if not matching_results:
            return SearchResult([], [])
            
        similar_titles = [
            {
                'title': result['title'],
                'score': result['score']
            }
            for result in matching_results[:8]
            if result['score'] > 0
        ]
        
        all_chunks = []
        for result in matching_results[:6]:
            if result['score'] > 0:
                all_chunks.extend(result['chunks'])
                
        return SearchResult(all_chunks, similar_titles)

    def save_index(self, directory: str):
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
        self.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        with open(os.path.join(directory, "index_data.pkl"), 'rb') as f:
            other_data = pickle.load(f)
            
        self.chunk_data = other_data['chunk_data']
        self.titles = other_data['titles']
        self.title_to_chunks = other_data['title_to_chunks']
        self.bm25 = other_data['bm25']
        self.chunk_to_context = defaultdict(list, other_data['chunk_to_context'])

    def get_all_titles(self) -> List[str]:
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