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
from qdrant_client.http import models

logger = logging.getLogger(__name__)


def load_vector_components() -> tuple:
    """
    Загружает векторные модели и настраивает Qdrant клиент.
    Возвращает: (retriever_model, reranker_model, reranker_tokenizer, qdrant_client, collection_name)
    """
    settings = get_settings()
   
    device = settings.models.device

    retriever_model = SentenceTransformer(
        settings.models.full_retriver_path, 
        local_files_only=True,
        device=device
    )

    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        settings.models.full_reranker_path
    )
    reranker_model = reranker_model.to(device)
    reranker_model.eval()

    reranker_tokenizer = AutoTokenizer.from_pretrained(
        settings.models.full_reranker_path
    )

    # Инициализация Qdrant клиента
    qdrant_client = QdrantClient(
        host=settings.qdrant.host, 
        port=settings.qdrant.port
    )

    collection_name = settings.qdrant.collection

    return retriever_model, reranker_model, reranker_tokenizer, qdrant_client, collection_name


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


# def rerank_documents(query, model, tokenizer, documents, top_k=3):
    """
    :param query:
    :param model:
    :param tokenizer:
    :param documents:
    :param top_k:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    if not documents:
        return []

    pairs = [[query, doc["candidate"]] for doc in documents]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        if model.config.num_labels == 1:
            scores = logits.squeeze(1)
        elif model.config.num_labels == 2:
            scores = logits[:, 1]
        else:
            raise ValueError(f"Неподдерживаемое число меток: {model.config.num_labels}")

    scored_docs = list(zip(documents, scores.cpu().numpy()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Время, затраченное на реранк: {str(time.time() - start_time)}")
    return [doc for doc, score in scored_docs[:top_k]]


def retrieve_docs(query, retriever_model, reranker_model, reranker_tokenizer, qdrant_client, collection_name, top_k=3):
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
    # return rerank_documents(query, reranker_model, reranker_tokenizer, initial_candidates, top_k=top_k)
    return initial_candidates