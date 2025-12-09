import asyncio
import logging
from contextlib import asynccontextmanager

import faiss
import json
from datetime import datetime
import torch.cuda

from app.config.config import PROJECT_DIR, LOGS_DIR
from fastapi import APIRouter, FastAPI
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel

from app.src.rag.search import retrieve_docs
from app.src.new.generate import generate_answer
from app.src.rag.search import load_vector_components
from app.src.sql.db import load_sql_db
from app.src.agent.setup import initialize_agent_with_tools

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Загрузка векторных компонентов")
    retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents = load_vector_components()

    logger.info("Загрузка базы данных")
    db = load_sql_db()

    logger.info("Инициализация агента")
    agent = initialize_agent_with_tools(
        retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents, db
    )

    yield


@router.post('/generate/', response_class=PlainTextResponse)
async def processing_message(request: QueryRequest):
    global agent
    if agent is None:
        return "Ошибка. Агент не загружен"

    query = f'Сегодня {datetime.now().strftime("%d %B %Y года")}. Вопрос: {request.query}'

    answer = agent.run(query)
    return answer
    # answer = await asyncio.to_thread(generate_answer, query, docs)
    # return answer
