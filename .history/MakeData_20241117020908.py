import pandas as pd
import glob
import os

def merge_all_csv_files(directory_path):
    """
    디렉토리 내의 모든 CSV 파일들을 통합합니다.
    
    Args:
        directory_path (str): CSV 파일들이 있는 디렉토리 경로
    
    Returns:
        pandas.DataFrame: 통합된 데이터프레임
    """
    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    
    # 빈 리스트로 시작하여 각 데이터프레임을 저장
    dfs = []
    
    print(f"Found {len(csv_files)} CSV files")
    
    # 각 CSV 파일을 읽어서 리스트에 추가
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # 파일명을 source_file 컬럼으로 추가
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
            print(f"Successfully read: {file} - {len(df)} rows")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # 모든 데이터프레임을 하나로 통합
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 결과 저장
    output_path = os.path.join(directory_path, 'merged_all.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged file saved as: {output_path}")
    print(f"Total rows: {len(merged_df)}")
    
    return merged_df

# 사용 예시
directory = "processed_files"  # 실제 디렉토리 경로로 변경하세요
merged_data = merge_all_csv_files(directory)