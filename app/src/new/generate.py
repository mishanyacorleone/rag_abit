from openai import OpenAI
from app.config.config import OPENROUTER_TOKEN, PROXY_URL, SYSTEM_PROMPT
import httpx


def generate_answer(query, docs):
    user_prompt = f"""
    Ответь на данный вопрос:
    <question>{query}</question>
    Проанализируй на основе предложенных документов и пришли человекочитаемый ответ.
    Если в документе содержится ответ из базы FAQ, то ты не должен сильно его переделывать.
    Документы:
    <documents>{docs}</documents>
    """

    http_client = httpx.Client(proxy=PROXY_URL)  # Опционально

    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=f"{OPENROUTER_TOKEN}",
      http_client=http_client  # Опционально
    )

    completion = client.chat.completions.create(
        extra_body={},
        model="tngtech/deepseek-r1t2-chimera:free",
        messages=[{
            "role": "system",
            "content": [{
                "type": "text",
                "text": f"{SYSTEM_PROMPT}"
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
