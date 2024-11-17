from typing import List, Dict, Any
from openai import OpenAI
import os
from search.FAISSIndex import search_content
from utils import MAX_CONTEXT_LENGTH

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

PROMPT_TEMPLATE = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.

Now use the following context items to answer the user query:
{context}

User query: {query}
Answer:"""

def generate_prompt(query: str, chunks: List[Dict[str, Any]], k: int = 10) -> str:
    relevant_chunks = search_content(query, chunks, k)

    context = ""
    total_length = 0
    
    for chunk in relevant_chunks:
        chunk_text = f"[{chunk['sentence_chunk']}]\n"
        chunk_length = len(chunk_text)
        
        if total_length + chunk_length > MAX_CONTEXT_LENGTH:
            break
        
        context += chunk_text
        total_length += chunk_length
    
    print(context)
    return PROMPT_TEMPLATE.format(context=context.strip(), query=query)

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about book information."},
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