import json
import logging
from uuid import uuid4
from typing import List, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, ScrollRequest
)
from sentence_transformers import SentenceTransformer

from app.config.config import get_settings

logger = logging.getLogger(__name__)


class QdrantManager:
    def __init__(self, retriever_model: SentenceTransformer):
        settings = get_settings()
        self.client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
        self.collection_name = settings.qdrant.collection
        self.model = retriever_model

    def ensure_collection_exists(self):
        """
        Создает коллекцию, если её нет
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            logger.info(f"Коллекция {self.collection_name} создана")
            return True
        return False

    def get_stats(self) -> dict:
        """
        Статистика коллекции
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "points_count": info.points_count,
                "status": info.status.value
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> dict:
        """
        Получить все документы с пагинацией
        """
        results, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        documents = []
        for point in results:
            documents.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"}
            })
        
        return {
            "documents": documents,
            "count": len(documents),
            "next_offset": next_offset
        }
    
    def search_documents(self, query: str, limit: int = 5) -> list:
        """
        Поиск документов по запросу
        """
        embeddings = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embeddings,
            limit=limit
        )

        print(type(results))
        print(results)

        return [
            {
                "id": str(r.id),
                "score": round(r.score, 4),
                "text": r.payload.get("text", "")
            }
            for r in results.points
        ]
    
    def add_document(self, text: str, metadata: dict = None):
        """
        Добавить ещё один документ
        """
        doc_id = str(uuid4())
        embedding = self.model.encode(text).tolist()

        payload = {"text": text}
        if metadata:
            payload.update(metadata)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=doc_id, vector=embedding, payload=payload)]
        )

        return doc_id
    
    def add_document_batch(self, documents: List[dict]) -> List[str]:
        """
        Добавить пакет документов
        documents: [{"text": "...", "metadata": {...}}, ...]
        """
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        points = []
        ids = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid4())
            payload = {"text": doc["text"]}
            if doc.get("metadata"):
                payload.update(doc["metadata"])
            
            points.append(PointStruct(id=doc_id, vector=embedding.tolist(), payload=payload))
            ids.append(doc_id)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points   
        )

        logger.info(f"Добавлено {len(ids)} документов")
        return ids
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Удалить документ по ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
            logger.info(f"Документ удалён: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления документа: {str(e)}")
            return False
        
    def delete_by_text(self, substring: str) -> int:
        """
        Удалить все документы, содержащие подстроку
        """
        all_docs, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        ids_to_delete = [
            point.id for point in all_docs
            if substring.lower() in point.payload.get("text", "").lower()
        ]

        if ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids_to_delete
            )

        logger.info(f"Удалено {len(ids_to_delete)} документов в подстроке: '{substring}'")
        return len(ids_to_delete)
    
    def clear_collection(self):
        """
        Очистить и пересоздать коллекцию
        """
        self.client.delete_collection(self.collection_name)
        self.ensure_collection_exists()
        logger.info("Коллекция очищена и пересоздана")

    def load_from_json(self, json_path: str) -> List[str]:
        """
        Загрузить документы из JSON файла
        """
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        documents = []
        for item in data:
            doc = {"text": item["text"]}
            metadata = {k: v for k, v in item.items() if k != "text"}
            if metadata:
                doc["metadata"] = metadata
            documents.append(doc)

        return self.add_document_batch(documents)
    
    def reload_from_json(self, json_path: str) -> List[str]:
        """
        Полная перезагрузка: очистка + загрузка из документа
        """
        self.clear_collection()
        return self.load_from_json(json_path)

