import os
import fitz
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import re
import functools

def text_formatter(text):
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    
    pdf_id = os.path.basename(pdf_path).replace('.pdf', '')
    
    for page_number, page in tqdm(enumerate(doc), desc=f"Processing {pdf_id}"):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "pdf_id": pdf_id,
            "page_number": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    return pages_and_texts

def process_pages(pages_and_texts, nlp, slice_size=10):
    for item in tqdm(pages_and_texts, desc="Processing pages"):
        item["sentences"] = [str(sentence) for sentence in nlp(item["text"]).sents]
        item["page_sentence_count_spacy"] = len(item["sentences"])
        item["sentence_chunks"] = [item["sentences"][i:i + slice_size] for i in range(0, len(item["sentences"]), slice_size)]
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

def create_chunks(pages_and_texts):
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Creating chunks"):
        for sentence_chunk in item["sentence_chunks"]:
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', "".join(sentence_chunk).replace("  ", " ").strip())
            chunk_dict = {
                "pdf_id": item["pdf_id"],
                "page_number": item["page_number"],
                "sentence_chunk": joined_sentence_chunk,
                "chunk_char_count": len(joined_sentence_chunk),
                "chunk_word_count": len(joined_sentence_chunk.split(" ")),
                "chunk_token_count": len(joined_sentence_chunk) / 4
            }
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

@functools.lru_cache(maxsize=1000)
def get_embedding(text, model):
    """Cache embeddings for identical text chunks"""
    return model.encode(text)

def create_embeddings(pages_and_chunks, model):
    for item in tqdm(pages_and_chunks, desc="Creating embeddings"):
        item["embedding"] = get_embedding(item["sentence_chunk"], model)
    return pages_and_chunks

def process_pdf_directory(pdf_dir, output_path, model_name="all-mpnet-base-v2", device="cpu"):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return None
    
    nlp = English()
    nlp.add_pipe("sentencizer")
    embedding_model = SentenceTransformer(model_name, device=device).to(device)
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n{pdf_file} 처리 중")
        
        pages_and_texts = process_pdf(pdf_path)
        pages_and_texts = process_pages(pages_and_texts, nlp)
        pages_and_chunks = create_chunks(pages_and_texts)
        pages_and_chunks = create_embeddings(pages_and_chunks, embedding_model)
        
        output_filename = f"{os.path.splitext(pdf_file)[0]}_processed.csv"
        output_filepath = os.path.join(output_path, output_filename)
        
        df = pd.DataFrame(pages_and_chunks)
        df.to_csv(output_filepath, index=False)
        
        print(f"\n완료, {output_filepath}에 저장되어 있음")

if __name__ == "__main__":
    pdf_directory = "pdf_files"  
    output_directory = "processed_files" 
    
    for directory in [pdf_directory, output_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    process_pdf_directory(
        pdf_dir=pdf_directory,
        output_path=output_directory,
        device="cpu"
    )