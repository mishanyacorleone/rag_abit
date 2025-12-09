from langchain_community.utilities import SQLDatabase
from pathlib import Path

def load_sql_db() -> SQLDatabase:
    """
    Загружает и возвращает объект SQLDatabase.

    :param uri: URI базы данных (по умолчанию 'sqlite:///data/university.db').
    :return: Объект SQLDatabase.
    """
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "university.db"
    # print(db_path)
    # print(f"------ {Path(__file__)} ------")
    uri = f"sqlite:///{db_path}"
    db = SQLDatabase.from_uri(uri)
    return db