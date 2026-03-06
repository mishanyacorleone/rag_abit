"""
Инициализация PostgreSQL
Загружает все csv файлы из data/psql/ через API приложения

Использует существующий эндпоинт /postgres/upload-csv
поэтому приложение должно быть запущено

Запуск:
    python scripts/init_psql.py

Или из контейнера:
    docker-compose exec app python scripts/init_psql.py
"""
import os
import sys
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8080")
CSV_DATA_DIR = os.getenv("PSQL_DATA_DIR", "data/psql")
UPLOAD_ENPOINT = f"{APP_BASE_URL}/api/admin/postgres/upload-csv"


def upload_csv(csv_path: Path) -> dict:
    """
    Загружает один CSV файл через API
    """
    table_name = csv_path.stem

    with open(csv_path, "rb") as f:
        response = requests.post(
            UPLOAD_ENPOINT,
            files={"file": (csv_path.name, f, "text/csv")},
            params={"table_name": table_name, "mode": "replace"},
            timeout=120
        )

    if response.status_code == 200:
        result = response.json()
        logger.info(
            f"{csv_path.name} -> '{table_name}': "
            f"{result.get('rows_loaded', '?')} строк, "
            f"столбцы: {result.get('columns', [])}"
        )
        return result
    else:
        logger.error(
            f"{csv_path.name}: {response.status_code} - {response.text}"
        )
        return {"error": response.text}


def main():
    data_path = Path(CSV_DATA_DIR)
    if not data_path.exists():
        logger.error(f"Директория не найдена: {data_path}")
        sys.exit(1)

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        logger.error(f"CSV файлы не найдены в {data_path}")
        sys.exit(1)

    logger.info(f"Найдено {len(csv_files)} CSV файлов в {data_path}")
    logger.info(f"API: {UPLOAD_ENPOINT}")
    logger.info("=" * 60)

    try:
        health = requests.get(f"{APP_BASE_URL}/health", timeout=5)
        logger.info(f"Приложение доступно (status={health.status_code})")
    except requests.ConnectionError:
        logger.error(
            f"Приложение не доступно по адресу {APP_BASE_URL}\n"
            f"Убедитесь, что docker-compose up выполнен и app запущен"
        )
        sys.exit(1)

    results = {}
    for csv_file in csv_files:
        logger.info(f"Загрузка: {csv_file.name}...")
        results[csv_file.name] = upload_csv(csv_file)

    logger.info("=" * 60)
    success = sum(1 for r in results.values() if "error" not in r)
    logger.info(f"Загружно: {success}/{len(csv_files)} файлов")


if __name__ == "__main__":
    main()