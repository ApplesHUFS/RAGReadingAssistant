import pandas as pd
import glob
import os
import re

def normalize_book_title(title):
    title = re.sub(r'^[A-Z]+-\d+-', '', title).replace("_", " ")
    
    parts = title.split('-')
    
    if len(parts) >= 2:
        author = parts[0]
        book_title = ' '.join(parts[1:-1]) if len(parts) > 2 else parts[1]
        return f"{book_title} {author}"
    
    return title 

def merge_all_csv_files(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    
    dfs = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        dfs.append(df)
        print(f"Successfully read: {file} - {len(df)} rows")
    
    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df['pdf_id'] = merged_df['pdf_id'].apply(normalize_book_title)
    
    merged_df.to_csv('book_data.csv', index=False)
    print(f"Total rows: {len(merged_df)}")
    return merged_df

directory = "processed_files"
merged_data = merge_all_csv_files(directory)