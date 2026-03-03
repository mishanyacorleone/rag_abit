from openai import OpenAI
from app.config.config import get_settings, SYSTEM_PROMPT_TEXT
import httpx
import json
from typing import List, Dict
import uuid


def generate_answer(query, docs):
    user_prompt = f"""
    Ответь на данный вопрос:
    <question>{query}</question>
    Проанализируй на основе предложенных документов и пришли человекочитаемый ответ.
    Если в документе содержится ответ из базы FAQ, то ты не должен сильно его переделывать.
    Документы:
    <documents>{docs}</documents>
    """
    settings = get_settings()

    client = OpenAI(
        base_url=settings.vllm.base_url,
        api_key=settings.vllm.api_key  # vLLM не требует ключа 
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=settings.vllm.model_name,
        messages=[{
            "role": "system",
            "content": [{
                "type": "text",
                "text": f"{SYSTEM_PROMPT_TEXT}"
            }]
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"{user_prompt}"
            }]
        }]
    )
    return completion.choices[0].message.content
