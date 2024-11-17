import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GPTHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))
        self.max_context_length = 5000
        
    def _create_prompt(self, book_title: str, query: str, contexts: List[Dict]) -> str:
        """GPT 프롬프트를 생성합니다."""
        context_text = ""
        total_length = 0
        
        for ctx in contexts:
            chunk_text = f"[관련도: {ctx['score']:.2f}] {ctx['chunk']}\n"
            if total_length + len(chunk_text) > self.max_context_length:
                break
            context_text += chunk_text
            total_length += len(chunk_text)
            
        return f"""You are a reading assistant for the book '{book_title}'. 
        Based on the following passages from the book, please provide a helpful and insightful answer to the user's question.
        Make sure your answers are explanatory and directly related to the book's content.
        If the context doesn't contain enough information to answer the question, please say so.

        Relevant passages from the book:
        {context_text}

        User's question: {query}
        Answer:"""
        
    def get_answer(self, book_title: str, query: str, contexts: List[Dict]) -> str:
        """GPT를 사용하여 답변을 생성합니다."""
        if not contexts:
            return "죄송합니다. 책에서 관련된 내용을 찾을 수 없습니다."
            
        prompt = self._create_prompt(book_title, query, contexts)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful reading assistant focused on providing insights about the current book."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"