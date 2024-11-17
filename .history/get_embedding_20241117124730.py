import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict

class TextProcessor:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
                 device: str = "cpu", 
                 max_tokens: int = 128):
        self.model = SentenceTransformer(model_name, device=device)
        self.tokenizer = self.model.tokenizer
        self.max_tokens = max_tokens

    def split_sentences(self, text: str) -> List[str]:
        pattern = r'(?<=[.!?])\s+(?=[^0-9])|(?<=[.!?])$'
        sentences = re.split(pattern, text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def create_adaptive_chunks(self, sentences: List[str]) -> List[List[str]]:
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                sub_sentences = re.split('[.,!?]', sentence)
                temp_chunk = []
                temp_tokens = 0
                
                for sub_sent in sub_sentences:
                    if not sub_sent.strip():
                        continue
                        
                    sub_tokens = self.count_tokens(sub_sent)
                    if temp_tokens + sub_tokens <= self.max_tokens:
                        temp_chunk.append(sub_sent)
                        temp_tokens += sub_tokens
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = [sub_sent]
                        temp_tokens = sub_tokens
                
                if temp_chunk:
                    chunks.append(temp_chunk)
                    
            elif current_tokens + sentence_tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(768)
        return self.model.encode(text)
            
def process_text_file(file_path: str, processor: TextProcessor) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text = re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()
    sentences = processor.split_sentences(text)
    chunks = processor.create_adaptive_chunks(sentences)
    
    pages_and_chunks = []
    for chunk_sentences in tqdm(chunks):
        joined_chunk = " ".join(chunk_sentences).strip()
        
        embedding = processor.get_embedding(joined_chunk)
        
        chunk_dict = {
            "pdf_id": os.path.basename(file_path).replace('.txt', ''),
            "page_number": 0,
            "sentence_chunk": joined_chunk,
            "chunk_char_count": len(joined_chunk),
            "chunk_word_count": len(processor.tokenizer.tokenize(joined_chunk)),
            "chunk_token_count": processor.count_tokens(joined_chunk),
            "num_sentences": len(chunk_sentences),
            "embedding": embedding
        }
        pages_and_chunks.append(chunk_dict)
    
    return pages_and_chunks

def process_text_directory(txt_dir: str, 
                         output_path: str, 
                         model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                         device: str = "cpu",
                         max_tokens: int = 128):
        
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    processor = TextProcessor(
        model_name=model_name,
        device=device,
        max_tokens=max_tokens
    )
    
    for txt_file in txt_files:
        print(f"\n{txt_file} 처리 중...")
        txt_path = os.path.join(txt_dir, txt_file)
        
        chunks_with_embeddings = process_text_file(txt_path, processor)
        
        output_filename = f"{os.path.splitext(txt_file)[0]}_processed.csv"
        output_filepath = os.path.join(output_path, output_filename)
        
        df = pd.DataFrame(chunks_with_embeddings)
        df['embedding'] = df['embedding'].apply(lambda x: ','.join(map(str, x)))
        
        df.to_csv(output_filepath, index=False)
        print(f"완료: {output_filepath}")
        
        print(f"총 청크 수: {len(chunks_with_embeddings)}")
        print(f"평균 토큰 수: {df['chunk_token_count'].mean():.1f}")
        print(f"최대 토큰 수: {df['chunk_token_count'].max()}")
        
if __name__ == "__main__":
    text_directory = "text_files"
    output_directory = "processed_files"
    
    for directory in [text_directory, output_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    process_text_directory(
        txt_dir=text_directory,
        output_path=output_directory,
        max_tokens=128,
        device="cpu"
    )