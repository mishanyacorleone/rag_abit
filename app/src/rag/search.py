import json
import os
from pathlib import Path
import time
import logging

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config.config import get_settings
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def load_vector_components() -> tuple:
    """
    Загружает векторные модели и настраивает Qdrant клиент.
    Возвращает: (retriever_model, reranker_model, reranker_tokenizer, qdrant_client, collection_name)
    """
    settings = get_settings()
   
    device = settings.models.device

    retriever_model = SentenceTransformer(
        settings.models.retriever_path, 
        local_files_only=True,
        device=device
    )

    # Инициализация Qdrant клиента
    qdrant_client = QdrantClient(
        host=settings.qdrant.host, 
        port=settings.qdrant.port,
    )

    collection_name = settings.qdrant.collection

    return retriever_model, qdrant_client, collection_name


def retrieve_from_index(query, qdrant_client, collection_name, retriever_model, top_k=10):
    """
    :param query:
    :param qdrant_client:
    :param collection_name:
    :param retriever_model:
    :param top_k:
    :return:
    """
    start_time = time.time()
    query_embedding = retriever_model.encode(query).tolist()

    # Поиск в Qdrant
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    ).points

    results = []
    for result in search_results:
        results.append({
            "score": float(result.score),
            "corpus_id": result.id,
            "candidate": result.payload.get("text", "")
        })

    time_to_retrieve = time.time() - start_time
    logging.info(f"Время, затраченное на первичный поиск: {time_to_retrieve}")

    return results


def retrieve_docs(query, retriever_model, qdrant_client, collection_name, top_k=3):
    """
    :param query:
    :param retriever_model:
    :param reranker_model:
    :param reranker_tokenizer:
    :param qdrant_client:
    :param collection_name:
    :param top_k:
    :return:
    """
    initial_candidates = retrieve_from_index(query, qdrant_client, collection_name, retriever_model, top_k=5)
    return initial_candidates