from typing import List, Dict
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from tqdm import tqdm

from config import settings
from core.exceptions import SummaryError

class SummaryHandler:
    def __init__(self):
        try:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(settings.SUMMARY_TOKENIZER)
            self.model = BartForConditionalGeneration.from_pretrained(settings.SUMMARY_MODEL)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        except Exception as e:
            raise SummaryError(f"Error initializing summary handler: {e}")
            
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))
        
    def _create_sections(self, chunks: List[str]) -> List[str]:
        sections = []
        current_section = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            
            if not current_section:
                current_section.append(chunk)
                current_tokens = chunk_tokens
                continue
                
            if current_tokens + chunk_tokens <= settings.MAX_INPUT_TOKENS - 100:
                current_section.append(chunk)
                current_tokens += chunk_tokens
            else:
                sections.append(" ".join(current_section))
                current_section = [current_section[-1], chunk]
                current_tokens = self._count_tokens(" ".join(current_section))
                
        if current_section:
            sections.append(" ".join(current_section))
            
        return sections
        
    def _summarize_section(self, text: str) -> str:
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                max_length=settings.MAX_INPUT_TOKENS,
                truncation=True,
                padding=True
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=settings.MAX_SUMMARY_TOKENS,
                    min_length=settings.MIN_SUMMARY_TOKENS,
                    num_beams=5,
                    length_penalty=2.0,
                    no_repeat_ngram_size=4,
                    early_stopping=True
                )
                
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.replace("</s>", "").replace("<s>", "").strip()
            
        except Exception as e:
            raise SummaryError(f"Error summarizing section: {e}")
            
    def generate_summary(self, chunks: List[str]) -> Dict:
        try:
            if not chunks:
                return {
                    'final_summary': "입력된 텍스트가 없습니다.",
                    'section_summaries': [],
                    'section_count': 0
                }
                
            sections = self._create_sections(chunks)
            section_summaries = []
            
            for section in tqdm(sections):
                summary = self._summarize_section(section)
                section_summaries.append(summary)
                
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
            
        except Exception as e:
            raise SummaryError(f"Error generating summary: {e}")