from dataclasses import dataclass
from datetime import datetime

@dataclass
class BookMetadata:
    file_name: str
    processed_date: str
    chunk_count: int
    file_hash: str

    @classmethod
    def create(cls, file_name: str, chunk_count: int, file_hash: str):
        return cls(
            file_name=file_name,
            processed_date=datetime.now().isoformat(),
            chunk_count=chunk_count,
            file_hash=file_hash
        )
