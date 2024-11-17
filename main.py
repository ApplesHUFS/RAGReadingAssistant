import os
from typing import Optional
from processor import BookProcessor
from searcher import BookSearcher
from gpt_handler import GPTHandler
from summary_handler import SummaryHandler
from dotenv import load_dotenv

load_dotenv()

class BookAssistant:
    def __init__(self):
        self.processor = BookProcessor()
        self.searcher = BookSearcher()
        self.gpt_handler = GPTHandler()
        self.summary_handler = SummaryHandler()
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
            
        while True:
            print("\n=== 작업 선택 ===")
            print("1. 질문하기")
            print("2. 요약하기")
            print("3. 이전 메뉴로 돌아가기")
            
            mode_choice = input("\n작업을 선택하세요 (1-3): ")
            
            if mode_choice == '1':
                self.searcher.load_book(book_id)
                self._chat_mode(book_id)
            elif mode_choice == '2':
                self._summary_mode(book_id)
            elif mode_choice == '3':
                break
            else:
                print("\n잘못된 선택입니다. 다시 선택해주세요.")
    
    def _chat_mode(self, book_id: str):
        """선택된 파일에 대한 대화 모드를 시작합니다."""
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
                    
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")
                
    def _summary_mode(self, book_id: str):
        """선택된 소설의 요약 모드를 실행합니다."""
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
            
        except Exception as e:
            print(f"요약 생성 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    assistant = BookAssistant()
    assistant.run()