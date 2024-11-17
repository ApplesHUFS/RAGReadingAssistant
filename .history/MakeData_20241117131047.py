import pandas as pd
import glob
import os
import re

def normalize_book_title(title):
    """
    책 제목을 '책 제목 - 저자' 형식으로 정규화하는 함수
    예: NT-2764-이무영-죄와벌 -> 죄와벌 - 이무영
    """
    # 파일 번호 패턴 제거 (NT-xxxx-, OT-xxx- 등)
    title = re.sub(r'^[A-Z]+-\d+-', '', title)
    
    # 하이픈으로 분리
    parts = title.split('-')
    
    if len(parts) >= 2:
        author = parts[0]
        # 마지막 부분이 출판사/연재처인 경우 제외
        book_title = '-'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
        # 괄호 안의 내용(권수 등) 유지
        return f"{book_title} - {author}"
    
    return title  # 분리할 수 없는 경우 원본 반환

def merge_all_csv_files(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    
    dfs = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        dfs.append(df)
        print(f"Successfully read: {file} - {len(df)} rows")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # pdf_id 정규화
    merged_df['pdf_id'] = merged_df['pdf_id'].apply(normalize_book_title)
    
    # 결과 저장
    merged_df.to_csv('book_data_normalized.csv', index=False)
    print(f"\nMerged file saved with normalized titles")
    print(f"Total rows: {len(merged_df)}")
    
    # 정규화된 제목 목록 출력
    unique_titles = merged_df['pdf_id'].unique()
    print("\nUnique normalized titles:")
    for title in sorted(unique_titles):
        print(title)
    
    return merged_df

directory = "processed_files"
merged_data = merge_all_csv_files(directory)