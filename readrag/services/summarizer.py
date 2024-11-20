from typing import List, Dict
import time
from datetime import datetime, timedelta
from collections import deque
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

from readrag.config import settings
from readrag.core.exceptions import SummaryError

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

class SummaryHandler:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=openai_api_key)
            self.encoding = tiktoken.encoding_for_model(settings.GPT_MODEL)
            
            self.request_times = deque(maxlen=500)
            self.daily_requests = deque(maxlen=10000)
            self.token_usage = deque(maxlen=200000)
            
        except Exception as e:
            raise SummaryError(f"Error initializing summary handler: {e}")
            
    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            raise SummaryError(f"Error counting tokens: {e}")
            
    def _check_rate_limits(self, tokens_to_use: int):
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)
        day_ago = current_time - timedelta(days=1)
        
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
        while self.daily_requests and self.daily_requests[0] < day_ago:
            self.daily_requests.popleft()
        while self.token_usage and self.token_usage[0] < minute_ago:
            self.token_usage.popleft()
            
        if (len(self.request_times) >= settings.MAX_REQUESTS_PER_MINUTE or
            len(self.daily_requests) >= settings.MAX_REQUESTS_PER_DAY or
            len(self.token_usage) + tokens_to_use > settings.MAX_TOKENS_PER_MINUTE):
            
            sleep_time = 1.5
            if self.request_times:
                sleep_time = max(sleep_time, (minute_ago - self.request_times[0]).total_seconds())
            time.sleep(sleep_time)
            return self._check_rate_limits(tokens_to_use)
            
        self.request_times.append(current_time)
        self.daily_requests.append(current_time)
        self.token_usage.extend([current_time] * tokens_to_use)
            
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
                
            if current_tokens + chunk_tokens <= settings.MAX_INPUT_TOKENS - 1000:
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
            input_tokens = self._count_tokens(text)
            estimated_output_tokens = settings.MAX_SUMMARY_TOKENS
            total_tokens = input_tokens + estimated_output_tokens
            
            self._check_rate_limits(total_tokens)
            
            response = self.client.chat.completions.create(
                model=settings.GPT_MODEL,
                messages=[
                    {"role": "system", "content": "다음 텍스트를 요약해주세요. 핵심 내용과 주요 논점을 누락없이 포함하고, 문맥의 흐름을 자연스럽게 유지하며, 원문의 의도와 뉘앑스를 보존하면서 명확하고 간결한 문장으로 작성해주세요."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=settings.MAX_SUMMARY_TOKENS,
                top_p=1,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
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
            
            for section in tqdm(sections, desc="Summarizing sections"):
                summary = self._summarize_section(section)
                section_summaries.append(summary)
                
            if len(section_summaries) == 1:
                final_summary = section_summaries[0]
            else:
                combined_summary = "\n\n".join(section_summaries)
                self._check_rate_limits(self._count_tokens(combined_summary))
                
                response = self.client.chat.completions.create(
                    model=settings.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "다음은 여러 섹션의 요약들입니다. 이들을 하나의 통합된 요약으로 만들어주세요. 전체적인 흐름을 자연스럽게 연결하고 중복되는 내용을 제거하여 통합해주세요."},
                        {"role": "user", "content": combined_summary}
                    ],
                    temperature=0.3,
                    max_tokens=settings.MAX_SUMMARY_TOKENS,
                    top_p=1,
                    frequency_penalty=0.3,
                    presence_penalty=0.3
                )
                
                final_summary = response.choices[0].message.content.strip()
                
            return {
                'final_summary': final_summary,
                'section_summaries': section_summaries,
                'section_count': len(sections)
            }
            
        except Exception as e:
            raise SummaryError(f"Error generating summary: {e}")