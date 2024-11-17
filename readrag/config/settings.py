from pathlib import Path

class Settings:
    # Get project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Directory settings
    DATA_DIR = PROJECT_ROOT / "data"
    BOOKS_DIR = DATA_DIR / "books"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    BOOKS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    # Model settings
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    SUMMARY_MODEL = 'gogamza/kobart-summarization'
    SUMMARY_TOKENIZER = 'gogamza/kobart-base-v2'
    
    # File settings
    ALLOWED_EXTENSIONS = {'.txt'}  # 확장자 제한
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB 제한
    
    # Processor settings
    MAX_TOKENS = 128
    OVERLAP_TOKENS = 20
    
    # Search settings
    SEARCH_K = 20
    SEARCH_THRESHOLD = 0.35
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 128
    
    # Summary settings
    MAX_INPUT_TOKENS = 1024
    MAX_SUMMARY_TOKENS = 128
    MIN_SUMMARY_TOKENS = 32
    OVERLAP_SUMMARY_TOKENS = 50