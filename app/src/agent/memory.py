import logging
from typing import List, Optional
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text
from app.config.config import get_settings

logger = logging.getLogger(__name__)


class ChatMemoryManager:
    """
    Хранит всю историю сообщений в PostgreSQL
    В контекстное окно модели отправляется только first_n + last_n пар.
    """

    def __init__(self, first_n: int = 2, last_n: int = 3):
        """
        first_n: сколько первых пар отправлять в контекстное окно
        last_n: сколько последних пар отправлять в контекстное окно
        """
        settings = get_settings()
        self.engine = create_engine(settings.db.uri)
        self.first_n = first_n
        self.last_n = last_n
        self._ensure_table()
    
    def _ensure_table(self):
        """
        Создает таблицу, если не существует
        """
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chat_user_time
                ON chat_history (user_id, created_at)
            """))

            conn.commit()
        
        logger.info("Таблица chat_history готова")
    
    # ==========================================
    #  Запись сообщений
    # ==========================================

    def add_message(self, user_id: str, role: str, content: str):
        """
        Добавить сообщение
        """
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_history (user_id, role, content)
                    VALUES (:user_id, :role, :content)
                """),
                {"user_id": user_id, "role": role, "content": content}
            )
            conn.commit()


    # ==========================================
    #  Контекстное окно для модели
    # ==========================================

    def get_context_window(self, user_id: str) -> str:
        """
        Возвращает контекстное окно: first_n + last_n пар

        Если сообщений мало - возвращает все
        """
        messages = self._get_user_messages(user_id)

        if not messages:
            return ""
        
        pairs = self._group_into_pairs(messages)
        total_pairs = len(pairs)

        if total_pairs <= self.first_n + self.last_n:
            return self._format_pairs(pairs, has_gap=False)

        first_pairs = pairs[:self.first_n]
        last_pairs = pairs[-self.last_n:]

        skipped = total_pairs - self.first_n - self.last_n
        
        return self._format_pairs(
            first_pairs, last_pairs,
            has_gap=True, skipped=skipped
        )
    
    def _get_user_messages(self, user_id: str) -> List[dict]:
        """
        Получить все сообщения пользователя
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT role, content
                    FROM chat_history
                    WHERE user_id = :user_id
                    ORDER BY created_at ASC
                """),
                {"user_id": user_id}
            )
            return [
                {"role": row[0], "content": row[1]}
                for row in result.fetchall()
            ]
        
    def _group_into_pairs(self, messages: List[dict]) -> List[dict]:
        """
        Группирует все сообщения в пары user+assistant
        """
        pairs = []
        i = 0
        while i < len(messages):
            pair = {"user": None, "assistant": None}

            if messages[i]["role"] == "user":
                pair["user"] = messages[i]["content"]
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    pair["assistant"] = messages[i + 1]["content"]
                    i += 2
                else:
                    i += 1
            else:
                pair["assistant"] = messages[i]["content"]
                i += 1
            
            pairs.append(pair)
        return pairs
    
    def _format_pairs(self, first_pairs: List[dict],
                      last_pairs: List[dict] = None,
                      has_gap: bool = False,
                      skipped: int = 0) -> str:
        """
        Форматирует пары в строку для промпта
        """
        parts = []

        if has_gap:
            parts.append("Начало диалога:")
            for pair in first_pairs:
                if pair["user"]:
                    parts.append(f"Пользователь: {pair["user"]}")
                if pair["assistant"]:
                    parts.append(f"Ассистент: {pair["assistant"]}")

            parts.append(f"\n... Пропущено {skipped} сообщений ...\n")
            parts.append(f"Недавние сообщения:")

            for pair in last_pairs:
                if pair["user"]:
                    parts.append(f"Пользователь: {pair["user"]}")
                if pair["assistant"]:
                    parts.append(f"Ассистент: {pair["assistant"]}")
        else:
            parts.append("История диалога:")
            for pair in first_pairs:
                if pair["user"]:
                    parts.append(f"Пользователь: {pair["user"]}")
                if pair["assistant"]:
                    parts.append(f"Ассистент: {pair["assistant"]}")
        print(parts)
        return "\n".join(parts)
    
    # ==========================================
    #  Полная история (для просмотра)
    # ==========================================

    def get_full_history(self, user_id: str) -> List[dict]:
        """
        Вся история пользователя
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT role, assistant, created_at
                    FROM chat_history
                    WHERE user_id = :user_id
                    ORDER_BY created_at ASC
                """),
                {"user_id": user_id}
            )

            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                }
                for row in result
            ]
        
    def get_stats(self) -> dict:
        """
        Общая статистика
        """
        with self.engine.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM chat_history")
            ).scalar()

            users_count = conn.execute(
                text("SELECT COUNT(DISTINCT user_id) FROM chat_history")
            ).scalar()

            today = conn.execute(
                text("""
                    SELECT COUNT(*) FROM chat_history
                    WHERE created_at >= CURRENT_DATE
                """)
            ).scalar()

            top_users = conn.execute(
                text("""
                    SELECT user_id, COUNT(*) as msg_count,
                                    MIN(created_at) as first_msg,
                                    MAX(created_at) as last_msg
                    FROM chat_history
                    GROUP BY user_id
                    ORDER BY msg_count DESC
                    LIMIT 10
                """)
            ).fetchall()

        return {
            "total_messages": total,
            "unique_users": users_count,
            "messages_today": today,
            "top_users": [
                {
                    "user_id": row[0],
                    "messages": row[1],
                    "first_message": str(row[2]),
                    "last_message": str(row[3])
                }
                for row in top_users
            ]
        }