import asyncio
import logging
from contextlib import asynccontextmanager

import faiss
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
import torch.cuda

from app.config.config import PROJECT_DIR, LOGS_DIR
from fastapi import APIRouter, FastAPI
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel

from app.src.rag.search import retrieve_docs
from app.src.rag.generate import generate_answer
from app.src.rag.search import load_vector_components
from app.src.sql.db import load_sql_db
from app.src.agent.setup import initialize_agent_with_tools
from app.src.agent.context_buffer import get_retrieved_context, clear_retrieved_context

from app.api.admin import init_managers, memory_mgr

router = APIRouter()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# === Определение глобальных переменных ===
agent = None

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None


def append_rag_data_to_json(query: str, response: str, retrieved_context: List[Dict[str, str]], 
                            json_file_path: str = "/mnt/mishutqa/PycharmProjects/abitBot/data/parse/ragchecker_data.json"):
    """
    Добавляет данные RAG запроса в JSON файл.
    
    Args:
        query (str): Вопрос пользователя
        response (str): Ответ, сгенерированный RAG системой
        retrieved_context (List[Dict[str, str]]): Контекст, извлеченный из базы знаний
        json_file_path (str): Путь к JSON файлу для сохранения данных
    """

    query_id = str(uuid.uuid4())
    entry = {
        "query_id": query_id,
        "query": query,
        "gt_answer": "",  # ground truth answer пока пустой, его нужно заполнять вручную
        "response": response,
        "retrieved_context": retrieved_context
    }

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    except FileNotFoundError:
        data = []

    data.append(entry)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Запись добавлена в {json_file_path} с ID: {query_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Загрузка векторных компонентов")
    retriever_model, reranker_model, reranker_tokenizer, qdrant_client, collection_name = load_vector_components()

    logger.info("Загрузка базы данных")
    db = load_sql_db()

    logger.info("Инициализация менеджеров данных")
    init_managers(retriever_model)

    logger.info("Инициализация агента")
    agent = initialize_agent_with_tools(
        retriever_model, reranker_model, reranker_tokenizer, qdrant_client, collection_name, db
    )

    yield


@router.post('/generate/', response_class=PlainTextResponse)
async def processing_message(request: QueryRequest):
    global agent
    from app.api.admin import memory_mgr

    if agent is None:
        return "Ошибка. Агент не загружен"

    user_id = request.user_id
    date_str = f'{datetime.now().strftime("%d %B %Y года")}'

    chat_context = memory_mgr.get_context_window(user_id)

    if chat_context:
        query = (
            f"Сегодня {date_str}.\n\n"
            f"{chat_context}\n\n"
            f"Новый вопрос от пользователя: {request.query}"
        )
    else:
        query = (
            f"Сегодня {date_str}.\n\n"
            f"Вопрос: {request.query}"
        )

    clear_retrieved_context()

    try:
        result = agent.invoke({'input': query})
        answer = result['output']
    except Exception as e:
        logger.error(f"Ошибка агента: {e}")
        answer = "Произошла ошибка, попробуйте переформулировать."
    
    memory_mgr.add_message(user_id, "user", request.query)
    memory_mgr.add_message(user_id, "assistant", answer)

    # # answer = agent.run(query)
    # answer = agent.invoke({'input': query})['output']
    # retrieved_context = get_retrieved_context() or []

    # append_rag_data_to_json(
    #     query=query,
    #     response=answer,
    #     retrieved_context=retrieved_context
    # )

    return answer
    # answer = await asyncio.to_thread(generate_answer, query, docs)
    # return answer
