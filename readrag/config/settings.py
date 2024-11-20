from pathlib import Path

class Settings:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    DATA_DIR = PROJECT_ROOT / "data"
    BOOKS_DIR = DATA_DIR / "books"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    DATA_DIR.mkdir(exist_ok=True)
    BOOKS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    
    GPT_MODEL = "gpt-4o-mini" 
    
    ALLOWED_EXTENSIONS = {'.txt'}  
    MAX_FILE_SIZE = 10 * 1024 * 1024 
    
    MAX_TOKENS = 128
    OVERLAP_TOKENS = 20
    
    SEARCH_K = 20
    SEARCH_THRESHOLD = 0.35
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 128
    
    MAX_INPUT_TOKENS = 12000 
    MAX_SUMMARY_TOKENS = 1000  
    MIN_SUMMARY_TOKENS = 100 
    OVERLAP_SUMMARY_TOKENS = 200 
    
    MAX_REQUESTS_PER_MINUTE = 450 
    MAX_REQUESTS_PER_DAY = 9500   
    MAX_TOKENS_PER_MINUTE = 180000 