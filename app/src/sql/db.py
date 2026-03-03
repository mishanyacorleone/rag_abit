from langchain_community.utilities import SQLDatabase
from app.config.config import get_settings


def load_sql_db() -> SQLDatabase:
    """
    Загружает и возвращает объект SQLDatabase для PostgreSQL.
    Настройки подключения берутся из .env файла в корне проекта.
    """
    
    settings = get_settings()

    db = SQLDatabase.from_uri(
        settings.db.uri,
        sample_rows_in_table_info=3,
        include_tables=settings.db.include_tables
    )
    return db