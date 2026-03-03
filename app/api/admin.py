import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import json
import tempfile
import os

from app.src.qdrant.manager import QdrantManager
from app.src.sql.manager import PostgresManager
from app.src.agent.memory import ChatMemoryManager

router = APIRouter(prefix="/admin", tags=["Администрирование"])
logger = logging.getLogger(__name__)

qdrant_mgr: Optional[QdrantManager] = None
postgres_mgr: Optional[PostgresManager] = None
memory_mgr: Optional[ChatMemoryManager] = None


def init_managers(retriever_model=None):
    global qdrant_mgr, postgres_mgr, memory_mgr
    qdrant_mgr = QdrantManager(retriever_model)
    qdrant_mgr.ensure_collection_exists()
    postgres_mgr = PostgresManager()
    memory_mgr = ChatMemoryManager(first_n=2, last_n=3)
    logger.info("Менеджеры данных инициализированы")


# ==========================================
#  Pydantic модели для Qdrant
# ==========================================

class AddDocumentRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None

    class Config: 
        json_schema_extra = {
            "example": {
                "text": "Приём документов начинается 20 июня 2026 года",
                "metadata": {"source": "правила_приема_2026"}
            }
        }


class AddDocumentBatchRequest(BaseModel):
    documents: List[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {"text": "Документ1", "metadata": {"source": "file1"}},
                    {"text": "Документ2", "metadata": {"source": "file2"}}
                ]
            }
        }


class DeleteByTextRequest(BaseModel):
    substring: str

    class Config:
        json_schema_extara = {
            "example": {
                "substring": "Правила приема 2024"
            }
        }


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class LoadJsonPathRequest(BaseModel):
    json_path: str

    class Config:
        json_schema_extra = {
            "example": {
                "json_path": "path/to/your/json/file.json"
            }
        }


# ==========================================
#  Pydantic модели для PostgreSQL
# ==========================================

class AddRowRequest(BaseModel):
    """
    Добавление строки в таблицу
    """
    row_data: dict

    class Config:
        json_schema_extra = {
            "example": {
                "row_data": {
                    "spec_name": "Прикладная информатика",
                    "year": 2025,
                    "min_score": 180,
                    "budget_places": 25
                }
            }
        }


# ==========================================
#  Qdrant: Статистика
# ==========================================

@router.get("/qdrant/stats", summary="Статистика коллекции")
def qdrant_stats():
    """
    Количество документов, статус коллекции
    """
    return qdrant_mgr.get_stats()


@router.get("/qdrant/documents", summary="Все документы")
def qdrant_get_documents(limit: int = 20, offset: int = 0):
    """
    Получить документы с пагинацией

    - limit: сколько документов вернуть (макс 100)
    - offset: смещение для пагинации
    """
    if limit > 100:
        limit = 100
    return qdrant_mgr.get_all_documents(limit=limit, offset=offset)


@router.post("/qdrant/search", summary="Поиск документов")
def qdrant_search(request: SearchRequest):
    """
    Семантический поиск по документам
    """
    results = qdrant_mgr.search_documents(request.query, request.limit)
    return {"query": request.query, "results": results}


# ==========================================
#  Добавление документов
# ==========================================

@router.post("/qdrant/documents", summary="Добавить один документ")
def qdrant_add_document(request: AddDocumentRequest):
    """
    Добавить в коллекцию один документ
    """
    doc_id = qdrant_mgr.add_document(request.text, request.metadata)
    return {
        "status": "ok",
        "doc_id": doc_id,
        "message": "Документ добавлен"
    }


@router.post("/qdrant/documents/batch", summary="Добавить пакет документов")
def qdrant_add_batch(request: AddDocumentBatchRequest):
    """
    Добавить несколько документов за 1 запрос

    Формат: [{"text": "...", "metadata": {...}}, ...]
    """
    ids = qdrant_mgr.add_document_batch(request.documents)
    return {
        "status": "ok",
        "count": len(ids),
        "doc_ids": ids,
        "message": f"Добавлено {len(ids)} документов"
    }


@router.post("/qdrant/documents/upload", summary="Загрузить JSON файл")
async def qdrant_upload_json(file: UploadFile = File(...)):
    """
    Загрузить документы из JSON файла.
    
    Формат файла: [{"text": "..."}, {"text": "...", "source": "..."}, ...
    """

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате .json")
    
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Невалидный JSON")
    
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON должен содержать массив объектов")
    
    for i, item in enumerate(data):
        if "text" not in item:
            raise HTTPException(
                status_code=400,
                detail=f"Объект {i} не содержит поле 'text'"
            )
    
    documents = []
    for item in data:
        doc = {"text": item["text"]}
        metadata = {k: v for k, v in item.items() if k != "text"}
        if metadata:
            doc["metadata"] = metadata
        documents.append(doc)
    
    ids = qdrant_mgr.add_document_batch(documents)
    return {
        "status": "ok",
        "filename": file.filename,
        "count": len(ids),
        "message": f"Загружено {len(ids)} документов из {file.filename}"
    }


@router.post("/qdrant/documents/load-path", summary="Загрузить JSON по пути на сервере")
def qdrant_load_json_path(request: LoadJsonPathRequest):
    """
    Загрузить документы из JSON файла по абсолютному пути на сервере
    """
    if not os.path.exists(request.json_path):
        raise HTTPException(status_code=404, detail=f"Файл не найден: {request.json_path}")
    
    ids = qdrant_mgr.load_from_json(request.json_path)
    return {
        "status": "ok",
        "path": request.json_path,
        "count": len(ids),
        "message": f"Загружено {len(ids)} документов"
    }


# ==========================================
#  Удаление документов
# ==========================================

@router.delete("/qdrant/documents/{doc_id}", summary="Удалить документ по ID")
def qdrant_delete_document(doc_id: str):
    """
    Удалить конкретный документ по его ID
    """
    success = qdrant_mgr.delete_document(doc_id)
    if success:
        return {"status": "ok", "deleted": doc_id}
    raise HTTPException(status_code=404, detail="Документ не найден")


@router.post("/qdrant/doucments/delete-by-text", summary="Удалить по подстроке")
def qdrant_delete_by_text(request: DeleteByTextRequest):
    """
    Удалить все документы, содержащие указанную подстроку
    """
    count = qdrant_mgr.delete_by_text(request.substring)
    return {
        "status": "ok",
        "deleted_count": count,
        "message": f"Удалено {count} документов с подстрокой {request.substring}"
    }


# ==========================================
#  Управление коллекцией
# ==========================================

@router.post("/qdrant/clear", summary="Очистить коллекцию")
def qdrant_clear():
    """
    Удалить все документы и пересоздать коллекцию
    """
    qdrant_mgr.clear_collection()
    return {
        "status": "ok",
        "message": "Коллекция очищена и пересоздана"
    }


@router.post("/qdrant/reload", summary="Перезагрузить из JSON")
def qdrant_reload(request: LoadJsonPathRequest):
    """
    Полная перезагрузка: очистить коллекцию и загрузить из JSON
    """
    if not os.path.exists(request.json_path):
        raise HTTPException(status_code=404, detail=f"Файл не найден: {request.json_path}")

    ids = qdrant_mgr.reload_from_json(request.json_path)
    return {
        "status": "ok",
        "count": len(ids),
        "message": f"Коллекция перезагружена: {len(ids)} документов"
    }


# ==========================================
#  PostgreSQL: Статистика
# ==========================================

@router.get("/postgres/stats", summary="Статистика БД")
def postgres_tables():
    """
    Все таблицы, столбцы, количество строк
    """
    return postgres_mgr.get_stats()


@router.get("/postgres/table/{table_name}", summary="Информация о таблице")
def postgres_table_info(table_name: str):
    """
    Столбцы, количество строк и превью первых 5 строк
    """
    try:
        return postgres_mgr.get_table_info(table_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==========================================
#  PostgreSQL: Загрузка CSV
# ==========================================

@router.post("/postgres/upload-csv", summary="Загрузить CSV в таблицу")
async def postgres_upload_csv(
        file: UploadFile = File(..., description="CSV файл"),
        table_name: Optional[str] = Query(
            None,
            description="Имя таблицы. Если не указано - берётся из имени файла"
        ),
        mode: str = Query(
            "replace",
            description="replace = перезаписать таблицу, append = добавить строки"
        )
):
    """
    Загрузить CSV файл в PostgreSQL

    - **Название таблицы** = название файла (без .csv), если не указано
    - **Название столбцов** = название колонок в CSV файле
    - **mode=replace** - пересоздаёт таблицу
    - **mode=append** - добавляет строки в существующую таблицу
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400,detail="Файл должен быть .csv")

    if mode not in ("replace", "append"):
        raise HTTPException(status_code=400, detail="mode должен быть 'replace' или 'append'")

    if not table_name:
        table_name = file.filename.replace(".csv", "")

    content = await file.read()

    try:
        result = postgres_mgr.load_csv_bytes(content, table_name, mode)
        return {"status_code": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
#  PostgreSQL: Удаление таблиц
# ==========================================

@router.delete("/postgres/table/{table_name}", summary="Удалить таблицу")
def postgres_drop_table(table_name: str):
    """
    Полностью удаляет таблицу из БД
    """
    try:
        postgres_mgr.drop_table(table_name)
        return {
            "status": "ok",
            "message": f"Таблица '{table_name}' удалена"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/postgres/table/{table_name}/rows", summary="Очистить таблицу")
def postgres_clear_table(table_name: str):
    """
    Удаляет все строки, но таблица остаётся
    """
    try:
        deleted = postgres_mgr.clear_table(table_name)
        return {
            "status": "ok",
            "table": table_name,
            "deleted_rows": deleted
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==========================================
#  PostgreSQL: Добавление строк
# ==========================================

@router.post("/postgres/table/{table_name}/row", summary="Добавить строку")
def postgres_add_row(table_name: str, request: AddRowRequest):
    """
    Добавить одну строку в таблицу

    Пример для таблицы "marks_last_years"
    ```json
    {
        "row_data": {
            "spec_name": "Прикладная информатика",
            "year": 2025,
            "min_score": 180
        }
    }
    ```
    """
    try:
        result = postgres_mgr.add_row(table_name, request.row_data)
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==========================================
#  История и аналитика
# ==========================================

@router.get("/chat/stats", summary="Статистика чатов")
def chat_stats():
    """
    Общая статистика по диалогам
    """
    return memory_mgr.get_stats()


# ==========================================
#  Пользовательский запрос
# ==========================================

@router.post("/postgres/query", summary="Запрос к таблице")
def postgres_query(query: str):
    """
    Пользователь указывает запрос и ему выдается результат
    """
    return postgres_mgr.sql_query(query)