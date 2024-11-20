from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from pathlib import Path
from urllib.parse import unquote

from readrag.config.settings import Settings
from readrag.services.processor import BookProcessor
from readrag.services.searcher import BookSearcher
from readrag.services.summarizer import SummaryHandler
from readrag.services.gpt import GPTHandler
from readrag.utils.file_handler import FileHandler

settings = Settings()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = BookProcessor()
searcher = BookSearcher()
summary_handler = SummaryHandler()
gpt_handler = GPTHandler()
file_handler = FileHandler(settings.PROCESSED_DIR)

class Query(BaseModel):
    question: str

@app.get("/api/books")
async def get_books():
    available_books = []
    for file in settings.BOOKS_DIR.glob("*.txt"):
        available_books.append({
            "name": file.name,
            "path": str(file),
            "size": file.stat().st_size
        })

    processed = processor.get_processed_files()
    
    return {
        "available": available_books,
        "processed": processed
    }

@app.post("/api/books/upload")
async def upload_book(file: UploadFile = File(...)):
    try:
        book_path = settings.BOOKS_DIR / file.filename
        with book_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/books/process/{filename}")
async def process_book(filename: str):
    try:
        filename = unquote(filename)  # URL 디코딩
        book_path = settings.BOOKS_DIR / filename
        if not book_path.exists():
            raise HTTPException(status_code=404, detail="Book not found")
        
        book_id = processor.process_file(str(book_path))
        return {"book_id": book_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/books/{book_id}/chat")
async def chat_with_book(book_id: str, query: Query):
    try:
        book_id = unquote(book_id)  # URL 디코딩
        searcher.load_book(book_id)
        results = searcher.search(query.question)
        if not results:
            return {"answer": "관련된 내용을 찾을 수 없습니다."}
            
        metadata = processor.get_book_metadata(book_id)
        answer = gpt_handler.get_answer(
            book_title=metadata.file_name,
            query=query.question,
            contexts=results
        )
        return {
            "answer": answer,
            "contexts": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/books/{book_id}/summary")
async def summarize_book(book_id: str):
    try:
        book_id = unquote(book_id)  # URL 디코딩
        chunks = processor.get_all_chunks(book_id)
        summary = summary_handler.generate_summary(chunks)
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))