import pandas as pd
import glob
import os

def merge_all_csv_files(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    dfs = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    
    output_path = os.path.join(directory_path, 'merged_all.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged file saved as: {output_path}")
    print(f"Total rows: {len(merged_df)}")
    
    return merged_df

directory = "processed_files"
merged_data = merge_all_csv_files(directory)