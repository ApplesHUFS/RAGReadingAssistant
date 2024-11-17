from pathlib import Path
from typing import List, Dict, Any

from config import settings
from core.exceptions import BookAssistantError
from services.processor import BookProcessor
from services.searcher import BookSearcher
from services.summarizer import SummaryHandler
from services.gpt import GPTHandler
from utils.file_handler import FileHandler

class CLI:
    def __init__(self):
        self.processor = BookProcessor()
        self.searcher = BookSearcher()
        self.summary_handler = SummaryHandler()
        self.gpt_handler = GPTHandler()
        self.file_handler = FileHandler(settings.PROCESSED_DIR)
        
    def run(self):
        """메인 애플리케이션 루프를 실행합니다."""
        while True:
            try:
                self._show_main_menu()
                choice = input("\n선택해주세요 (1-3): ")
                
                if choice == '3':
                    print("\n프로그램을 종료합니다. 감사합니다!")
                    break
                    
                elif choice == '1':
                    self._add_new_file()
                elif choice == '2':
                    self._select_file()
                else:
                    print("\n잘못된 선택입니다. 다시 선택해주세요.")
                    
            except BookAssistantError as e:
                print(f"\n오류가 발생했습니다: {e}")
            except Exception as e:
                print(f"\n예상치 못한 오류가 발생했습니다: {e}")
                
    def _show_main_menu(self):
        """메인 메뉴를 표시합니다."""
        print("\n=== 독서 도우미 시스템 ===")
        print("1. 새로운 텍스트 파일 추가")
        print("2. 기존 파일 선택")
        print("3. 종료")
        
    def _add_new_file(self):
        """새로운 텍스트 파일을 추가합니다."""
        # 파일 목록 표시
        print("\n=== 사용 가능한 텍스트 파일 ===")
        books = self.file_handler.get_available_books()
        
        if books:
            print("\n[기존 파일 목록]")
            for i, book_info in enumerate(books, 1):
                print(f"{i}. {book_info['name']} ({book_info['size'] / 1024:.1f}KB)")
                
        print("\n[옵션]")
        print("1. 새 파일 가져오기")
        print("2. 기존 파일 선택")
        print("3. 이전 메뉴로 돌아가기")
        
        choice = input("\n선택해주세요: ")
        
        if choice == '1':
            self._import_new_book()
        elif choice == '2' and books:
            self._select_existing_book(books)
        elif choice == '3':
            return
        else:
            print("\n잘못된 선택입니다.")
            
    def _import_new_book(self):
        """새 책 파일을 가져옵니다."""
        file_path = input("\n텍스트 파일 경로를 입력하세요: ")
        if not file_path:
            return
            
        try:
            file_path = Path(file_path)
            self.file_handler.validate_file(file_path)
            imported_path = self.file_handler.import_book(file_path)
            
            print(f"\n파일을 성공적으로 가져왔습니다: {imported_path.name}")
            process_now = input("지금 처리하시겠습니까? (y/n): ")
            
            if process_now.lower() == 'y':
                self._process_selected_file(imported_path)
                
        except Exception as e:
            print(f"\n오류: {e}")
            
    def _select_existing_book(self, books: List[Dict[str, Any]]):
        """기존 책 파일을 선택합니다."""
        try:
            idx = int(input("\n처리할 파일 번호를 선택하세요: ")) - 1
            if 0 <= idx < len(books):
                self._process_selected_file(books[idx]['path'])
            else:
                print("\n잘못된 선택입니다.")
        except ValueError:
            print("\n잘못된 입력입니다.")
            
    def _select_file(self):
        """기존 파일을 선택하고 작업 모드를 선택합니다."""
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
        
        while True:
            print("\n=== 작업 선택 ===")
            print("1. 질문하기")
            print("2. 요약하기")
            print("3. 이전 메뉴로 돌아가기")
            
            mode_choice = input("\n작업을 선택하세요 (1-3): ")
            
            if mode_choice == '1':
                self._chat_mode()
            elif mode_choice == '2':
                self._summary_mode()
            elif mode_choice == '3':
                break
            else:
                print("\n잘못된 선택입니다.")
                
    def _process_selected_file(self, file_path: Path):
        """선택된 파일을 처리합니다."""
        try:
            print("\n파일 처리를 시작합니다...")
            book_id = self.processor.process_file(str(file_path))
            print(f"\n파일 처리가 완료되었습니다. ID: {book_id}")
        except Exception as e:
            print(f"\n파일 처리 중 오류가 발생했습니다: {e}")
            
    def _chat_mode(self):
        """선택된 파일에 대한 대화 모드를 시작합니다."""
        book_id = self.searcher.get_current_book()
        if not book_id:
            print("선택된 책이 없습니다.")
            return
            
        metadata = self.processor.get_book_metadata(book_id)
        print(f"\n=== {metadata.file_name} 에 대해 질문해보세요 ===")
        
        while True:
            query = input("\n질문을 입력하세요 ('b': 이전 메뉴로 돌아가기, 'q': 종료): ")
            
            if query.lower() == 'q':
                print("\n프로그램을 종료합니다. 감사합니다!")
                exit()
            elif query.lower() == 'b':
                break
                
            try:
                print("\n관련 내용을 검색중...")
                results = self.searcher.search(query)
                
                if not results:
                    print("관련된 내용을 찾을 수 없습니다. 다른 질문을 해보세요.")
                    continue
                    
                print("\nGPT의 답변을 생성중...")
                answer = self.gpt_handler.get_answer(
                    book_title=metadata.file_name,
                    query=query,
                    contexts=results
                )
                
                print("\n=== 답변 ===")
                print(answer)
                print("\n=== 참고한 구절들 ===")
                for r in results:
                    print(f"\n[관련도: {r['score']:.2f}]")
                    print(r['chunk'])
                    
            except BookAssistantError as e:
                print(f"오류가 발생했습니다: {e}")
                
    def _summary_mode(self):
        """선택된 파일의 요약 모드를 실행합니다."""
        book_id = self.searcher.get_current_book()
        if not book_id:
            print("선택된 책이 없습니다.")
            return
            
        metadata = self.processor.get_book_metadata(book_id)
        print(f"\n=== {metadata.file_name} 요약을 시작합니다 ===")
        
        try:
            print("\n전체 텍스트를 로드하는 중...")
            chunks = self.processor.get_all_chunks(book_id)
            
            summary_result = self.summary_handler.generate_summary(chunks)
            
            print("\n=== 전체 요약 ===")
            print(summary_result['final_summary'])
            
            print(f"\n=== 상세 요약 (총 {summary_result['section_count']}개 섹션) ===")
            for i, section_summary in enumerate(summary_result['section_summaries'], 1):
                print(f"\n[섹션 {i}]")
                print(section_summary)
                
            input("\n메인 메뉴로 돌아가려면 Enter를 누르세요...")
            
        except BookAssistantError as e:
            print(f"요약 생성 중 오류가 발생했습니다: {e}")