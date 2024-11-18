import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
from readrag.config import settings
from readrag.core.exceptions import FileNotFoundError

class FileHandler:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def get_available_books(self) -> List[Dict[str, Any]]:
        """books 디렉토리의 모든 텍스트 파일 정보를 반환"""
        files = []
        for file_path in settings.BOOKS_DIR.glob('*'):
            if file_path.suffix in settings.ALLOWED_EXTENSIONS:
                files.append({
                    'path': file_path,
                    'name': file_path.name,
                    'size': file_path.stat().st_size
                })
        return files
        
    def validate_file(self, file_path: Path) -> None:
        """파일 유효성을 검사하고 문제가 있을 경우 예외 발생"""
        if not file_path.exists():
            raise FileNotFoundError("파일이 존재하지 않습니다.")
            
        if file_path.suffix not in settings.ALLOWED_EXTENSIONS:
            raise FileNotFoundError(f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(settings.ALLOWED_EXTENSIONS)}")
            
        if file_path.stat().st_size > settings.MAX_FILE_SIZE:
            max_size_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise FileNotFoundError(f"파일이 너무 큽니다. 최대 크기: {max_size_mb}MB")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
        except UnicodeDecodeError:
            raise FileNotFoundError("파일이 UTF-8 형식이 아닙니다.")
            
    def import_book(self, file_path: Path) -> Path:
        """파일을 books 디렉토리로 복사."""
        target_path = settings.BOOKS_DIR / file_path.name
        
        if target_path.exists():
            i = 1
            while True:
                new_name = f"{target_path.stem}_{i}{target_path.suffix}"
                target_path = settings.BOOKS_DIR / new_name
                if not target_path.exists():
                    break
                i += 1
                
        shutil.copy2(file_path, target_path)
        return target_path
        
    def calculate_file_hash(self, file_path: str) -> str:
        """파일의 MD5 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            raise FileNotFoundError(f"Error calculating file hash: {e}")
            
    def load_json(self, file_path: Path) -> Dict[str, Any]:
        """JSON 파일을 로드"""
        if not file_path.exists():
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Error loading JSON file: {e}")
            
    def save_json(self, file_path: Path, data: Dict[str, Any]):
        """데이터를 JSON 파일로 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise FileNotFoundError(f"Error saving JSON file: {e}")
            
    def read_text_file(self, file_path: str) -> str:
        """텍스트 파일 읽기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Error reading text file: {e}")