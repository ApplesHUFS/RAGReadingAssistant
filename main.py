from gpt_integration import get_gpt_answer
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import os

def load_single_file(file_path: str) -> List[Dict[str, Any]]:
    text_chunks_and_embedding_df = pd.read_csv(file_path)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return text_chunks_and_embedding_df.to_dict(orient="records")

def load_processed_data(processed_dir: str) -> List[Dict[str, Any]]:
    all_processed_data = []
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(processed_dir, csv_file)   #file_path = processed_dir + "/" + csv_file
        file_data = load_single_file(file_path)
        all_processed_data.extend(file_data)
    return all_processed_data

def chatting(query: str, processed_dir: str = "processed_files") -> str:
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

"""
RAG 파이프라인 실행 순서

1. main.py
    1-1. main() 함수 시작
    1-2. 사용자로부터 쿼리 입력
    1-3. chatting(query) 함수 호출
    1-4. load_processed_data(processed_dir)로 CSV에서 데이터베이스 로드하기
        디렉토리의 모든 파일들 대상으로 ~
        텍스트 청크와 임베딩이 포함된 CSV 파일 읽고-> .read_csv(file_path)
        embedding 컬럼을 numpy array로 변환하고 -> .apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        DataFrame을 dictionary 형태로 변환하기 (.to_dict(orient="records")) 

2. GPT 프로세스 (gpt_integration.py)
    2-1. get_gpt_answer(query, processed_data) 실행
    2-2. generate_prompt(query, chunks, k)
        search 함수로 관련 청크 검색하고 -> search(query, chunks, k=20)
        컨텍스트와 쿼리를 템플릿에 결합하고 -> PROMPT_TEMPLATE.format(context=context.strip(), query=query)
        컨텍스트 길이 제한 적용하기 -> if total_length + chunk_length > MAX_CONTEXT_LENGTH: break

3. 검색 프로세스 (search.py)
    - search(query, chunksm, k) 함수에서
        utils.py의 get_embedding(text)으로 쿼리 임베딩 생성하고 -> model.encode(text).tolist()
        각 청크와 쿼리 임베딩 간 코사인 유사도 계산하고 -> cosine_similarity(query_embedding, chunk['embedding'])
        RELEVANCE_THRESHOLD(0.35) 기준으로 청크 필터링하고 -> if score >= RELEVANCE_THRESHOLD
        유사도 점수로 정렬하고 -> sort(key=lambda x: x['relevance_score'], reverse=True)
        상위 k개 청크 반환하기 -> scored_results[:k]

4. 답변 생성 프로세스 (다시 gpt_integration.py)
    4-1. 검색된 컨텍스트로 최종 프롬프트 생성
    4-2. generate_answer(prompt) 함수에서
        GPT-3.5-turbo 모델로 API 호출 -> client.chat.completions.create(model="gpt-3.5-turbo", temperature=0.3)
        시스템 프롬프트 + 사용자 프롬프트 전송 -> messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        응답 받아서 처리 -> response.choices[0].message.content.strip()

5. 최종 출력 (main.py)
    5-1. GPT 응답을 사용자에게 반환
    5-2. 결과 출력
"""

"""
사용 모델 및 임계값 정리
    - 임베딩: sentence-transformers의 'all-mpnet-base-v2'
    - 제너레이터: GPT-3.5-turbo
    - 컨텍스트 제한: 5000 토큰
    - 유사도 임계값: 0.35

파일 역할 정리
    - main.py: 사용자 인터페이스, 데이터 로딩
    - gpt_integration.py: GPT 통합, 프롬프트 관리
    - search.py: 텍스트 청크 검색 로직
    - utils.py: 임베딩, 유사도 계산 등 유틸리티
"""



"""
1. 원하는 폴더로 이동하기
    cd "폴더경로"
    ex: cd "E:\2024\영대 학술제 준비\implementation\Code"

2. git 초기화하기
    git init

3. .gitignore 설정 (큰 파일 제외)

4. 파일 추가 및 커밋하기
    git add .
    git commit -m "feat: 초기 커밋"

5. main 브랜치 생성하기
    git branch -M main

6. 원격 저장소 연결하기
    git remote add origin https://github.com/사용자명/저장소이름.git
    예: git remote add origin https://github.com/ApplesHUFS/AcademicFestival.git

7. 푸시하기
git push -f origin main
"""