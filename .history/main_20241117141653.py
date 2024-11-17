from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
from search import search_title, faiss_index, get_all_titles
import pandas as pd
import numpy as np

def load_processed_data(file_path: str = "book_data.csv") -> List[Dict[str, Any]]:
    text_chunks_and_embedding_df = pd.read_csv(file_path)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip('[]').replace('\n', '').replace(',', ' '), sep=' ')
    )
    return text_chunks_and_embedding_df.to_dict(orient="records")

def chatting(query: str, chunks: List[Dict[str, Any]]) -> str:
    return get_gpt_answer(query, chunks)

def initialize_search(processed_data: List[Dict[str, Any]]):
    if faiss_index.index is None:
        faiss_index.build_index(processed_data)

def display_all_titles():
    titles = get_all_titles()
    print("\n현재 데이터베이스에 있는 책 목록:")
    for i, title in enumerate(sorted(titles), 1):
        print(f"{i}. {title}")

ㅋㅋ