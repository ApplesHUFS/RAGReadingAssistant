from typing import List, Dict, Any
from openai import OpenAI
import os
from search import search
from utils import MAX_CONTEXT_LENGTH

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

SYSTEM_PROMPT = """당신은 책의 내용에 대해 답변하는 도우미입니다.
다음과 같은 지침을 따라 답변해주세요:

1. 주어진 컨텍스트만을 기반으로 답변하세요.
2. 컨텍스트에서 찾을 수 없는 내용은 답변하지 마세요.
3. 답변은 자연스러운 대화체로 작성하되, 정확하고 구체적이어야 합니다.
4. 질문과 관련된 책의 구체적인 내용을 인용하면서 설명해주세요.
5. 불확실한 내용에 대해서는 "컨텍스트에서 해당 내용을 찾을 수 없습니다"라고 명시해주세요.
"""

PROMPT_TEMPLATE = """주어진 컨텍스트를 바탕으로 질문에 답변해주세요.

[컨텍스트]
{context}

[관련 도서 정보]
{book_titles}

[사용자 질문]
{query}

답변 작성 시 주의사항:
1. 컨텍스트에서 발견한 구체적인 내용을 바탕으로 답변하세요.
2. 책의 내용을 직접 인용할 때는 "책에서는 ~라고 설명합니다"와 같은 형식을 사용하세요.
3. 관련된 추가 설명이 있다면 "또한, ~"과 같은 형식으로 자연스럽게 이어주세요.
4. 답변 마지막에는 필요한 경우 "더 자세한 내용이 궁금하시다면 구체적으로 질문해 주세요"라고 안내해주세요.

답변:"""

def generate_prompt(query: str, chunks: List[Dict[str, Any]], k: int = 10) -> str:
    relevant_chunks = search(query, chunks, k)
    book_titles = set()
    context = ""
    total_length = 0
    
    for chunk in relevant_chunks:
        chunk_text = f"[{chunk['pdf_id']}]\n{chunk['sentence_chunk']}\n\n"
        chunk_length = len(chunk_text)
        
        if total_length + chunk_length > MAX_CONTEXT_LENGTH:
            break
            
        book_titles.add(chunk['pdf_id'])
        context += chunk_text
        total_length += chunk_length
    
    book_titles_text = "\n".join([f"- {title}" for title in sorted(book_titles)])
    
    return PROMPT_TEMPLATE.format(
        context=context.strip(),
        book_titles=book_titles_text,
        query=query
    )

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def get_gpt_answer(query: str, processed_data: List[Dict[str, Any]]) -> str:
    prompt = generate_prompt(query, processed_data)
    return generate_answer(prompt)