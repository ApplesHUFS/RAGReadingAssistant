from pathlib import Path
import hashlib
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict

@dataclass
class BookMetadata:
    file_name: str
    processed_date: str
    chunk_count: int
    file_hash: str

class BookProcessor:
    def __init__(self, base_dir: str = "processed_books"):
        self.base_dir = Path(base_dir)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.max_tokens = 128
        self._load_metadata()
        
    def _load_metadata(self):
        metadata_path = self.base_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = {k: BookMetadata(**v) for k, v in json.load(f).items()}
        else:
            self.metadata = {}
            
    def _save_metadata(self):
        with open(self.base_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({k: asdict(v) for k, v in self.metadata.items()}, f, indent=2, ensure_ascii=False)
            
    def _calculate_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def _create_chunks(self, text: str) -> List[str]:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            tokens = self.model.tokenizer.encode(sentence)
            if current_tokens + len(tokens) <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += len(tokens)
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = len(tokens)
                
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks
        
    def process_file(self, file_path: str) -> str:
        """새로운 텍스트 파일을 처리하고 인덱싱합니다."""
        file_path = Path(file_path)
        file_hash = self._calculate_file_hash(file_path)
        book_id = file_path.stem
        
        # 이미 처리된 파일인지 확인
        if book_id in self.metadata and self.metadata[book_id].file_hash == file_hash:
            print(f"파일 {book_id}는 이미 처리되어 있습니다.")
            return book_id
            
        # 책 디렉토리 생성
        book_dir = self.base_dir / book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        
        # 텍스트 처리
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        chunks = self._create_chunks(text)
        chunk_data = []
        embeddings = []
        
        print(f"총 {len(chunks)}개의 청크를 처리중...")
        for i, chunk in enumerate(tqdm(chunks)):
            embedding = self.model.encode(chunk, convert_to_numpy=True)
            chunk_data.append({
                'file_id': book_id,
                'chunk_id': i,
                'sentence_chunk': chunk,
                'chunk_char_count': len(chunk),
                'chunk_word_count': len(chunk.split()),
                'chunk_token_count': len(self.model.tokenizer.encode(chunk)),
                'num_sentences': len([s for s in chunk.split('.') if s.strip()])
            })
            embeddings.append(embedding)
            
        # 청크 데이터 저장
        pd.DataFrame(chunk_data).to_csv(book_dir / "chunks.csv", index=False, encoding='utf-8')
        
        # 임베딩 저장
        embeddings_array = np.array(embeddings)
        np.save(book_dir / "embeddings.npy", embeddings_array)
        
        # FAISS 인덱스 생성
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        faiss.write_index(index, str(book_dir / "faiss.index"))
        
        # 메타데이터 업데이트
        self.metadata[book_id] = BookMetadata(
            file_name=file_path.name,
            processed_date=datetime.now().isoformat(),
            chunk_count=len(chunks),
            file_hash=file_hash
        )
        self._save_metadata()
        
        return book_id

    def get_processed_files(self) -> Dict[str, BookMetadata]:
        """처리된 파일 목록을 반환합니다."""
        return self.metadata

    def get_book_metadata(self, book_id: str) -> BookMetadata:
        """특정 책의 메타데이터를 반환합니다."""
        return self.metadata.get(book_id)