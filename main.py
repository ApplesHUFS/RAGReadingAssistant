from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
from search import search_title, faiss_index, get_all_titles, save_index, load_index
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

def initialize_search(processed_data: List[Dict[str, Any]], index_dir: str = "saved_index"):
    if os.path.exists(os.path.join(index_dir, "faiss_index.bin")):
        load_index(index_dir)
        print("저장된 인덱스를 로드했습니다.")
    else:
        print("새로운 인덱스를 생성합니다...")
        faiss_index.build_index(processed_data)
        save_index(index_dir)

def display_all_titles():
    titles = get_all_titles()
    print("\n현재 데이터베이스에 있는 책 목록:")
    for i, title in enumerate(sorted(titles), 1):
        print(f"{i}. {title}")

def main():
    processed_data = load_processed_data()
    initialize_search(processed_data)
    
    while True:
        print("\n제목을 입력하세요 (전체 목록 보기: 'list', 종료: 'q'): ")
        title_query = input()
        
        if title_query.lower() == 'q':
            break
            
        if title_query.lower() == 'list':
            display_all_titles()
            continue
        
        matching_chunks = search_title(title_query)

        if not matching_chunks:
            print("죄송합니다. 검색하신 제목의 책을 찾을 수 없습니다.")
            print("비슷한 제목을 찾아보시겠어요? 'list' 를 입력하시면 전체 책 목록을 볼 수 있습니다.")
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