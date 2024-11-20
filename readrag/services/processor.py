from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from readrag.config import settings
from readrag.core.book_metadata import BookMetadata
from readrag.core.exceptions import ProcessingError
from readrag.utils.file_handler import FileHandler

class BookProcessor:
    def __init__(self):
        self.base_dir = settings.PROCESSED_DIR
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.file_handler = FileHandler(self.base_dir)
        self._load_metadata()
        
    def _load_metadata(self):
        metadata_path = self.base_dir / "metadata.json"
        data = self.file_handler.load_json(metadata_path)
        self.metadata = {k: BookMetadata(**v) for k, v in data.items()}
        
    def _save_metadata(self):
        metadata_path = self.base_dir / "metadata.json"
        data = {k: v.__dict__ for k, v in self.metadata.items()}
        self.file_handler.save_json(metadata_path, data)
        
    def _create_chunks(self, text: str) -> List[str]:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            tokens = self.model.tokenizer.encode(sentence)
            if current_tokens + len(tokens) <= settings.MAX_TOKENS:
                current_chunk.append(sentence)
                current_tokens += len(tokens)
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                    overlap_sentences = current_chunk[-1:]
                    current_chunk = overlap_sentences
                    current_tokens = sum(len(self.model.tokenizer.encode(s)) for s in overlap_sentences)
                else:
                    current_chunk = [sentence]
                    current_tokens = len(tokens)
                    
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks
        
    def process_file(self, file_path: str) -> str:
        try:
            file_path = Path(file_path)
            file_hash = self.file_handler.calculate_file_hash(file_path)
            book_id = file_path.stem
            
            if book_id in self.metadata and self.metadata[book_id].file_hash == file_hash:
                return book_id
                
            book_dir = self.base_dir / book_id
            book_dir.mkdir(parents=True, exist_ok=True)
            
            text = self.file_handler.read_text_file(file_path)
            chunks = self._create_chunks(text)
            
            chunk_data = []
            embeddings = []
            
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
                
            pd.DataFrame(chunk_data).to_csv(book_dir / "chunks.csv", index=False, encoding='utf-8')
            
            embeddings_array = np.array(embeddings)
            faiss.normalize_L2(embeddings_array)
            np.save(book_dir / "embeddings.npy", embeddings_array)
            
            dimension = embeddings_array.shape[1]
            index = faiss.IndexHNSWFlat(dimension, settings.HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = settings.HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = settings.HNSW_EF_SEARCH
            index.add(embeddings_array)
            
            faiss.write_index(index, str(book_dir / "faiss.index"))
            
            self.metadata[book_id] = BookMetadata.create(
                file_name=file_path.name,
                chunk_count=len(chunks),
                file_hash=file_hash
            )
            self._save_metadata()
            
            return book_id
            
        except Exception as e:
            raise ProcessingError(f"Error processing file: {e}")
            
    def get_processed_files(self) -> Dict[str, any]:
        return self.metadata
        
    def get_book_metadata(self, book_id: str) -> BookMetadata:
        return self.metadata.get(book_id)
        
    def get_all_chunks(self, book_id: str) -> List[str]:
        book_dir = self.base_dir / book_id
        if not book_dir.exists():
            raise ProcessingError(f"Book {book_id} not found")
            
        chunks_df = pd.read_csv(book_dir / "chunks.csv", encoding='utf-8')
        chunks_df = chunks_df.sort_values('chunk_id')
        return chunks_df['sentence_chunk'].tolist()