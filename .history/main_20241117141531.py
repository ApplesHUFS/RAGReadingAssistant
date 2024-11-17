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

def load_processed_data(file_path: str = "book_data.csv") -> List[Dict[str, Any]]:
    try:
        text_chunks_and_embedding_df = pd.read_csv(file_path)
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
            lambda x: np.fromstring(x.strip('[]').replace('\n', '').replace(',', ' '), sep=' ')
        )
        return text_chunks_and_embedding_df.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return []
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []

def initialize_search(processed_data: List[Dict[str, Any]]) -> bool:
    try:
        if not processed_data:
            print("No data loaded to initialize search")
            return False
            
        print("Initializing search index...")
        faiss_index.build_index(processed_data)
        print("Search index initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing search: {str(e)}")
        return False

def main():
    print("Loading data...")
    processed_data = load_processed_data()
    
    if not processed_data:
        print("Failed to load data. Exiting...")
        return
        
    if not initialize_search(processed_data):
        print("Failed to initialize search. Exiting...")
        return
    
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