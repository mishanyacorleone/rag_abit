import json
import os
from pathlib import Path
import time
import logging

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config.config import PROJECT_DIR
import faiss
logger = logging.getLogger(__name__)


def load_vector_components() -> tuple:
    """
    Загружает векторные модели, индекс и документы.
    Возвращает: (retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents)
    """
    MODELS_DIR = PROJECT_DIR / "app" / "models"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    retriever_model = SentenceTransformer(str(MODELS_DIR / "deepvk" / "USER-bge-m3"), local_files_only=True, device=device)

    reranker_model = AutoModelForSequenceClassification.from_pretrained(str(MODELS_DIR / "BAAI" / "bge-reranker-v2-m3"))
    reranker_model = reranker_model.to(device)
    reranker_model.eval()

    reranker_tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "BAAI" / "bge-reranker-v2-m3"))

    INDEX_PATH = PROJECT_DIR / "data" / "all_faiss_index.idx"
    faiss_index = faiss.read_index(str(INDEX_PATH))

    DOCUMENTS_PATH = PROJECT_DIR / "data" / "all_documents_meta.json"
    with open(str(DOCUMENTS_PATH), "r", encoding="utf-8") as file:
        documents = json.load(file)

    return retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents


def retrieve_from_index(query, index, texts, retriever_model, top_k=10):
    """
    :param query:
    :param index:
    :param texts:
    :param retriever_model:
    :param top_k:
    :return:
    """

    start_time = time.time()
    query_embedding = retriever_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "corpus_id": int(idx),
            "candidate": texts[idx]["text"]
        })

    time_to_retrieve = time.time() - start_time
    logging.info(f"Время, затраченное на первичный поиск: {time_to_retrieve}")

    return results


def rerank_documents(query, model, tokenizer, documents, top_k=3):
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


def retrieve_docs(query, retriever_model, reranker_model, reranker_tokenizer, index, all_documents, top_k=3):
    """
    :param query:
    :param retriever_model:
    :param reranker_model:
    :param index:
    :param all_documents:
    :param top_k:
    :return:
    """

    initial_candidates = retrieve_from_index(query, index, all_documents, retriever_model, top_k=10)
    return rerank_documents(query, reranker_model, reranker_tokenizer, initial_candidates, top_k=top_k)
    # logger.info(f"{initial_candidates}")
    # return initial_candidates


if __name__ == "__main__":
    query = "Какие проходные баллы на прикладную информатику?"
    retriever_model = SentenceTransformer(r"../../models/deepvk/USER-bge-m3", local_files_only=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(r"../../models/BAAI/bge-reranker-v2-m3")
    reranker_model.eval()
    reranker_tokenizer = AutoTokenizer.from_pretrained(r"../../models/BAAI/bge-reranker-v2-m3")

    PROJECT_PATH = Path(__file__).parent.parent.parent.parent

    INDEX_PATH = PROJECT_PATH / "data" / "all_faiss_index.idx"
    faiss_index = faiss.read_index(str(INDEX_PATH))

    DOCUMENTS_PATH = PROJECT_PATH / "data" / "all_documents_meta.json"
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
        docs = json.load(file)

    processed_docs = retrieve_docs(
        query,
        retriever_model,
        reranker_model,
        reranker_tokenizer,
        faiss_index,
        docs,
    )