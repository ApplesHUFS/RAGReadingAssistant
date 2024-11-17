from typing import List, Dict
import os
from openai import OpenAI
from dotenv import load_dotenv

from readrag.core.exceptions import ProcessingError

load_dotenv()

class GPTHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_context_length = 5000
        
    def _create_prompt(self, book_title: str, query: str, contexts: List[Dict]) -> str:
        context_text = ""
        total_length = 0
        
        for ctx in contexts:
            chunk_text = f"[관련도: {ctx['score']:.2f}] {ctx['chunk']}\n"
            if total_length + len(chunk_text) > self.max_context_length:
                break
            context_text += chunk_text
            total_length += len(chunk_text)
            
        return f"""당신은 '{book_title}'의 독서 도우미입니다. 다음과 같은 원칙에 따라 답변해 주세요:

1. 답변 방식:
   - 책의 내용을 기반으로 명확하고 이해하기 쉽게 설명해 주세요
   - 필요한 경우 책의 구절을 직접 인용하여 설명을 뒷받침해 주세요
   - 독자의 이해를 돕기 위해 필요한 맥락이나 배경 정보를 추가해 주세요

2. 답변 품질:
   - 단순한 요약이나 피상적인 설명은 피해 주세요
   - 책의 핵심 개념과 아이디어를 깊이 있게 분석해 주세요
   - 가능한 경우 실제 적용 방법이나 예시를 제시해 주세요

3. 제한사항:
   - 주어진 문맥에 없는 내용은 추측하지 말아 주세요
   - 정보가 불충분할 경우, 어떤 정보가 부족한지 구체적으로 설명해 주세요
   - 확실하지 않은 내용에 대해서는 그 불확실성을 명시해 주세요

다음은 책에서 관련된 구절들입니다:
{context_text}

질문: {query}

위 내용을 바탕으로 독자의 이해를 돕는 통찰력 있는 답변을 제공해 주세요:"""
        
    def get_answer(self, book_title: str, query: str, contexts: List[Dict]) -> str:
        try:
            if not contexts:
                return "죄송합니다. 책에서 관련된 내용을 찾을 수 없습니다. 다른 방식으로 질문을 해주시거나, 찾고자 하는 내용을 더 구체적으로 설명해 주시면 도움이 될 것 같습니다."
                
            prompt = self._create_prompt(book_title, query, contexts)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 책의 내용을 깊이 있게 이해하고 설명하는 독서 도우미입니다. 독자의 질문에 대해 책의 내용을 바탕으로 통찰력 있고 교육적인 답변을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise ProcessingError(f"Error generating GPT answer: {e}")