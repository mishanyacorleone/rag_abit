from pathlib import Path
from functools import lru_cache
from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)


PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()
PROMPTS_DIR = Path(__file__).parent / "prompts"
LOGS_DIR = PROJECT_DIR / "app" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_prompt(filename: str) -> str:
    """
    Загружает промпт из файла
    
    :param filename: Description
    :type filename: str
    :return: Description
    :rtype: str
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Промпт не найден: {path}")
    
    return path.read_text(encoding="utf-8").strip()


# ==========================================
#  Pydantic Settings
# ==========================================


class VLLMSettings(BaseSettings):
    """
    Настройки vLLM сервера
    """

    base_url: str = "http://localhost:2222/v1"
    model_name: str = "qwen3-abit"
    api_key: str = "EMPTY"

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


class QdrantSettings(BaseSettings):
    """
    Настройки Qdrant
    """
    host: str = "localhost"
    port: int = 6333
    collection: str = "documents"
    data_path: str = "data/qdrant_meta.json"

    @property
    def full_data_path(self) -> Path:
        return PROJECT_DIR / self.data_path
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


class DatabaseSettings(BaseSettings):
    """
    Настройки PostgreSQL
    """
    host: str = "localhost"
    port: int = 5432
    name: str = "abit_db"
    user: str = "postgres"
    password: str = "qwerty123"

    @property
    def uri(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    include_tables: list = [
        # "marks_last_years",
        # "spec_info",
        # "vi_soo_vo",
        # "vi_spo",
        # "prices"
    ]

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ModelSettings(BaseSettings):
    """
    Настройки локальных моделей
    """

    retriever_path: str = "app/models/USER-bge-m3"
    reranker_path: str = "app/models/bge-reranker-v2-m3"
    device: str = "cpu"
    
    @property
    def full_retriver_path(self) -> str:
        return str(PROJECT_DIR / self.retriever_path)
    
    @property
    def full_reranker_path(self) -> str:
        return str(PROJECT_DIR / self.reranker_path)
    
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class DataSettings(BaseSettings):
    """
    Настройки путей к данным
    """
    csv_dir: str = "data/csv"

    @property
    def full_csv_dir(self) -> Path:
        return PROJECT_DIR / self.csv_dir
    
    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


class AppSettings(BaseSettings):
    """
    Общие настройки приложения
    """

    host: str = "0.0.0.0"
    port: int = "8081"
    debug: bool = True
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

  
class Settings:
    """
    Конвейер, который собирает всё вместе
    """

    def __init__(self):
        self.app = AppSettings()
        self.vllm = VLLMSettings()
        self.qdrant = QdrantSettings()
        self.db = DatabaseSettings()
        self.models = ModelSettings()
        self.data = DataSettings()

    def print_config(self):
        """
        Выводит текущую конфигурацию
        """

        print("=" * 50)
        print("APP CONFIG")
        print("=" * 50)
        print(f"  App:    {self.app.host}:{self.app.port}")
        print(f"  vLLM:    {self.vllm.base_url} ({self.vllm.model_name})")
        print(f"  Qdrant:    {self.qdrant.host}:{self.qdrant.port}/{self.qdrant.collection}")
        print(f"  Postgres:    {self.db.host}:{self.db.port}/{self.db.name}")
        print(f"  Device:    {self.models.device}")
        print("=" * 50)


@lru_cache()
def get_settings() -> Settings:
    """
    Синглтон - создается 1 раз, потом переиспользуется.
    Кэшируется через lru_cache
    """        
    return Settings()


SYSTEM_PROMPT_TEXT = load_prompt("system_prompt.txt")
SQL2TEXT_PROMPT = load_prompt("sql2text_prompt.txt")
CUSTOM_AGENT_PROMPT = load_prompt("custom_agent_prompt.txt")
TABLE_SELECTOR_PROMPT = load_prompt("table_selector_prompt.txt")

CUSTOM_AGENT_PROMPT = PromptTemplate.from_template(CUSTOM_AGENT_PROMPT)
