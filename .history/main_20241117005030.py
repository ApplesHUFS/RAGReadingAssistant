from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import os

def load_processed_data(file_path: str = "book_data.csv") -> List[Dict[str, Any]]:
    text_chunks_and_embedding_df = pd.read_csv(file_path)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return text_chunks_and_embedding_df.to_dict(orient="records")

def chatting(query: str, processed_dir: str = "book_data.csv") -> str:
    processed_data = load_processed_data(processed_dir)
    return get_gpt_answer(query, processed_data)

def main():
    while True:
        query = input("\n쿼리를 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
        
        answer = chatting(query)
        print("\n답변:")
        print(answer)

if __name__ == "__main__":
    main()