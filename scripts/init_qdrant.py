"""
Инициализация Qdrant

1. Собирает все JSON из /app/data/qdrant
2. Преобразует их в единый формат:
   {
       "text": "...",
       "meta": ""
   }
3. Создает коллекцию
4. Загружает эмбеддинги в Qdrant

Запуск:
    docker compose exec app python scripts/init_qdrant.py
"""

import json
import os
import logging
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer


# ==============================
# Настройки
# ==============================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL_PATH = os.getenv(
    "MODEL_RETRIEVER_PATH",
    "/app/models/deepvk/USER-bge-m3"
)

DATA_FOLDER = Path("/app/data/qdrant")
VECTOR_SIZE = 1024
BATCH_SIZE = 64


# ==============================
# Обработчики файлов
# ==============================

def process_ind_achievements(file_path: Path):
    results = []
    data = json.load(open(file_path, encoding="utf-8"))

    for section in data.values():
        for item in section:
            text_parts = []

            if "Название индивидуального достижения" in item:
                text_parts.append(
                    f"Название индивидуального достижения: "
                    f"{item['Название индивидуального достижения']}"
                )

            if "Документы" in item:
                text_parts.append(f"Документы: {item['Документы']}")

            if "Балл" in item:
                text_parts.append(f"Балл: {item['Балл']}")

            text = ". ".join(text_parts)

            results.append({
                "text": text.strip(),
                "meta": ""
            })

    return results


def process_osobie_prava(file_path: Path):
    results = []
    data = json.load(open(file_path, encoding="utf-8"))

    for item in data:
        results.append({
            "text": item["text"],
            "meta": item.get("meta", "")
        })

    return results


def process_pravila_priema(file_path: Path):
    results = []
    data = json.load(open(file_path, encoding="utf-8"))

    for chapter, paragraphs in data.items():
        for paragraph in paragraphs:
            text = (
                f"Документ: Правила приема. "
                f"Глава {chapter}. {paragraph}"
            )
            results.append({
                "text": text.strip(),
                "meta": ""
            })

    return results


# ==============================
# Сбор всех JSON
# ==============================

def collect_all_json():
    all_chunks = []

    for file_path in DATA_FOLDER.glob("*.json"):
        if file_path.name == "qdrant_meta.json":
            continue

        logger.info(f"Обработка файла: {file_path.name}")

        if "ind_achievements" in file_path.name:
            chunks = process_ind_achievements(file_path)

        elif "osobie_prava" in file_path.name:
            chunks = process_osobie_prava(file_path)

        elif "pravila_priema" in file_path.name:
            chunks = process_pravila_priema(file_path)

        else:
            logger.warning(f"Нет обработчика для {file_path.name}")
            continue

        all_chunks.extend(chunks)

    logger.info(f"Всего собрано чанков: {len(all_chunks)}")

    with open(DATA_FOLDER / "qdrant_meta.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return all_chunks


# ==============================
# Работа с Qdrant
# ==============================

def create_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        logger.info("Удаляю старую коллекцию...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    logger.info(f"Коллекция '{COLLECTION_NAME}' создана")


def upload_to_qdrant(client, model, data):
    if not data:
        logger.warning("Нет данных для загрузки.")
        return

    texts = [item["text"] for item in data]

    logger.info("Кодирование текстов...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )

    points = []
    for item, embedding in zip(data, embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload=item
            )
        )

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Загружено {min(i+BATCH_SIZE, len(points))}/{len(points)}")

    logger.info("Загрузка завершена ✅")


# ==============================
# main
# ==============================

def main():
    logger.info("Загрузка модели эмбеддингов...")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    logger.info("Подключение к Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    logger.info("Сбор JSON...")
    data = collect_all_json()

    create_collection(client)
    upload_to_qdrant(client, model, data)


if __name__ == "__main__":
    main()