import os
from typing import Optional
from processor import BookProcessor
from searcher import BookSearcher
from gpt_handler import GPTHandler
from dotenv import load_dotenv

load_dotenv()

class BookAssistant:
    def __init__(self):
        self.processor = BookProcessor()
        self.searcher = BookSearcher()
        self.gpt_handler = GPTHandler()
        os.makedirs("processed_books", exist_ok=True)
        
    def run(self):
        """메인 애플리케이션 루프를 실행합니다."""
        while True:
            self._show_main_menu()
            choice = input("선택해주세요 (1-3): ")
            
            if choice == '1':
                self._add_new_file()
            elif choice == '2':
                self._select_file()
            elif choice == '3':
                print("\n프로그램을 종료합니다. 감사합니다!")
                break
            else:
                print("\n잘못된 선택입니다. 다시 선택해주세요.")
                
    def _show_main_menu(self):
        """메인 메뉴를 표시합니다."""
        print("\n=== 독서 도우미 시스템 ===")
        print("1. 새로운 텍스트 파일 추가")
        print("2. 기존 파일 선택")
        print("3. 종료")
                
    def _add_new_file(self):
        """새로운 텍스트 파일을 추가합니다."""
        file_path = input("\n텍스트 파일 경로를 입력하세요: ")
        if not os.path.exists(file_path):
            print("파일을 찾을 수 없습니다.")
            return
            
        try:
            print("\n파일 처리를 시작합니다...")
            book_id = self.processor.process_file(file_path)
            print(f"\n파일 처리가 완료되었습니다. ID: {book_id}")
        except Exception as e:
            print(f"파일 처리 중 오류가 발생했습니다: {e}")
            
    def _select_file(self):
        """기존 파일을 선택합니다."""
        metadata = self.processor.get_processed_files()
        if not metadata:
            print("\n처리된 파일이 없습니다. 새 파일을 추가해주세요.")
            return
            
        print("\n=== 처리된 파일 목록 ===")
        for i, (book_id, meta) in enumerate(metadata.items(), 1):
            print(f"{i}. {meta.file_name}")
            print(f"   처리일시: {meta.processed_date}")
            print(f"   청크 수: {meta.chunk_count}")
            
        try:
            choice = int(input("\n파일을 선택하세요 (번호): "))
            book_id = list(metadata.keys())[choice - 1]
        except (ValueError, IndexError):
            print("잘못된 선택입니다.")
            return
            
        self.searcher.load_book(book_id)
        self._chat_mode(book_id)
        
    def _chat_mode(self, book_id: str):
        """선택된 파일에 대한 대화 모드를 시작합니다."""
        metadata = self.processor.get_book_metadata(book_id)