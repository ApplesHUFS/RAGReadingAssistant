from typing import List, Dict
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # transformers의 경고 메시지 숨김

class SummaryHandler:
    def __init__(self):
        """한국어 소설 요약을 위한 모델 초기화"""
        model_path = "gogamza/kobart-summarization"
        
        # 토크나이저와 모델 초기화
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"모델을 {self.device}에서 실행합니다.")
        
        # 토큰 제한 설정
        self.max_input_tokens = 1024  # KoBART 입력 제한
        self.max_summary_tokens = 128  # 요약 길이 제한
        self.min_summary_tokens = 32   # 최소 요약 길이
        self.overlap_tokens = 50  # 문맥 유지를 위한 오버랩
        
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산"""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
        
    def _create_sections(self, chunks: List[str]) -> List[str]:
        """청크들을 입력 제한을 고려하여 섹션으로 분할"""
        sections = []
        current_section = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            
            # 현재 섹션이 비어있는 경우
            if not current_section:
                current_section.append(chunk)
                current_tokens = chunk_tokens
                continue
                
            # 청크를 추가해도 제한을 넘지 않는 경우
            if current_tokens + chunk_tokens <= self.max_input_tokens - 100:  # 여유 공간 확보
                current_section.append(chunk)
                current_tokens += chunk_tokens
            else:
                # 현재 섹션 저장하고 새로운 섹션 시작
                sections.append(" ".join(current_section))
                # 문맥 유지를 위해 마지막 청크를 다음 섹션의 시작으로 사용
                current_section = [current_section[-1], chunk]
                current_tokens = self._count_tokens(" ".join(current_section))
        
        # 마지막 섹션 처리
        if current_section:
            sections.append(" ".join(current_section))
            
        return sections
        
    def _summarize_section(self, text: str) -> str:
        """개별 섹션 요약"""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_input_tokens, 
                              truncation=True, padding=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_summary_tokens,
                min_length=self.min_summary_tokens,
                num_beams=5,
                length_penalty=2.0,
                no_repeat_ngram_size=4,
                early_stopping=True
            )
            
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.replace("</s>", "").replace("<s>", "").strip()
        
    def generate_summary(self, chunks: List[str]) -> Dict:
        """소설 전체 요약 생성"""
        if not chunks:
            return {
                'final_summary': "입력된 텍스트가 없습니다.",
                'section_summaries': [],
                'section_count': 0
            }
            
        print("텍스트를 섹션으로 분할하는 중...")
        sections = self._create_sections(chunks)
        
        print(f"총 {len(sections)}개의 섹션을 요약하는 중...")
        section_summaries = []
        for i, section in enumerate(tqdm(sections)):
            summary = self._summarize_section(section)
            section_summaries.append(summary)
        
        # 섹션 요약들을 결합하여 최종 요약 생성
        print("최종 요약을 생성하는 중...")
        if len(section_summaries) == 1:
            final_summary = section_summaries[0]
        else:
            combined_summary = " ".join(section_summaries)
            final_summary = self._summarize_section(combined_summary)
        
        return {
            'final_summary': final_summary,
            'section_summaries': section_summaries,
            'section_count': len(sections)
        }