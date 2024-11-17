# main.py 수정
from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
from search import search_title, search
import pandas as pd
import numpy as np
import os

def load_processed_data(file_path: str = "book_data.csv") -> List[Dict[str, Any]]:
    text_chunks_and_embedding_df = pd.read_csv(file_path)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip('[]').replace('\n', '').replace(',', ' '), sep=' ')
    )
    return text_chunks_and_embedding_df.to_dict(orient="records")

def chatting(query: str, chunks: List[Dict[str, Any]]) -> str:
    return get_gpt_answer(query, chunks)

def main():
    processed_data = load_processed_data()
    
    while True:
        print("\n제목을 입력하세요 (종료하려면 'q' 입력): ")
        title_query = input()
        
        if title_query.lower() == 'q':
            break

        matching_chunks = search_title(title_query)
        
        if not matching_chunks:
            print("죄송합니다. 그 제목을 가진 책은 데이터베이스에 없어요")
            continue
        
        print("\n그 책을 읽다가 궁금했던 점들을 입력하세요 (다른 책을 검색하려면 'b', 종료하려면 'q' 입력):")
        
        while True:
            content_query = input()
            
            if content_query.lower() == 'q':
                return
            if content_query.lower() == 'b':
                break
            
            answer = chatting(content_query, matching_chunks)
            print("\n답변:")
            print(answer)

if __name__ == "__main__":
    main()