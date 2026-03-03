import re
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import text

from app.config.config import SQL2TEXT_PROMPT

logger = logging.getLogger(__name__)


def sql_query_only(query: str, db: SQLDatabase, llm) -> dict:
    """
    Генерирует SQL-запрос и выполняет его.
    Возвращает сырые данные без форматирования через LLM.

    Возвращает:
        {
            "success": True/False,
            "query": "SELECT ...",
            "data": [{"col1": "val1", ...}, ...],
            "error": None или текст ошибки
        }
    """

    prompt = ChatPromptTemplate.from_template(SQL2TEXT_PROMPT)

    def get_schema(_):
        return db.get_table_info()
    
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

    generated_query = None

    try:
        generated_query_raw = sql_chain.invoke({"question": query})
        generated_query = extract_sql_query(generated_query_raw)
        logger.info(f"Сгенерированный SQL: {generated_query}")

        engine = db._engine
        with engine.connect() as conn:
            result_proxy = conn.execute(text(generated_query))
            column_names = list(result_proxy.keys())
            rows = result_proxy.fetchall()

            data = [dict(zip(column_names, row)) for row in rows]

        logger.info(f"SQL вернул: {len(data)} строк")

        return {
            "success": True,
            "query": query,
            "data": data,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Ошибка SQL: {e}")
        return {
            "success": False,
            "query": query,
            "data": [],
            "error": str(e)
        }
    

def extract_sql_query(text: str) -> str:
    """
    Извлекает SQL-запрос из сгенерированного текста модели.
    Игнорирует посторонний текст до начала SQL.
    """
    # 1. Попробуем найти SQL в markdown-блоке
    pattern = re.compile(r'```(?:sql)?\s*\n?(.*?)\n?```', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        extracted = match.group(1).strip()
        # Проверим, начинается ли результат с SQL-команды
        if looks_like_sql(extracted.split()[0]) if extracted.split() else False:
            return extracted

    # 2. Ищем начало SQL в тексте
    lines = text.strip().splitlines()
    sql_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Ищем строку, начинающуюся с ключевого слова SQL
        if re.match(r'^(select|with|insert|update|delete|from|where|limit|union|order|group|having|join|alter|drop|create)', stripped, re.IGNORECASE):
            sql_start = i
            break

    if sql_start != -1:
        # Объединяем строки с начала SQL
        sql_part = "\n".join(lines[sql_start:]).strip()
        # Убираем лишние строки после SQL
        sql_lines = []
        for line in sql_part.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if looks_like_sql(stripped):
                sql_lines.append(line)
            else:
                break
        result = "\n".join(sql_lines).strip()
        # Убедимся, что результат начинается с SQL-команды
        if result and looks_like_sql(result.split()[0]):
            return result

    # 3. Если ничего не нашли — возвращаем исходный текст (на всякий случай)
    return text.strip()


def looks_like_sql(token: str) -> bool:
    """
    Проверяет, похоже ли начало строки на SQL-запрос.
    """
    token_lower = token.lower()
    sql_keywords = [
        'select', 'with', 'insert', 'update', 'delete', 'from', 'where', 'group', 'order', 'limit', 'union', 'having', 'join', 'on', 'and', 'or', 'not', 'like', 'as',
        '(', '"', "'"
    ]
    for kw in sql_keywords:
        if token_lower.startswith(kw):
            return True
    return False





















# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.utilities import SQLDatabase
# from langchain_core.runnables import RunnablePassthrough
# from app.config.config import SQL2TEXT_PROMPT, SYSTEM_PROMPT_TEXT
# from app.src.agent.context_buffer import set_retrieved_context
# from datetime import datetime
# from typing import Optional, List, Dict
# import re
# import uuid
# import json


# def sql_search(query: str, db: SQLDatabase, llm) -> str:
#     """
#     Генерирует и выполняет SQL-запрос к базе данных.

#     :param query: Вопрос пользователя.
#     :param db: Объект SQLDatabase.
#     :param llm: Объект LLM (например, ChatOpenAI).
#     :return: Результат выполнения SQL-запроса.
#     """
#     prompt = ChatPromptTemplate.from_template(SQL2TEXT_PROMPT)

#     def get_schema(_):
#         return db.get_table_info()

#     sql_response = (
#         RunnablePassthrough.assign(schema=get_schema)
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     # --- Генерация запроса ---
#     generated_query_raw = sql_response.invoke({"question": query})

#     generated_query = extract_sql_query(generated_query_raw)

#     # --- Выполнение запроса ---
#     print(generated_query)
#     # raw_result = db.run(generated_query)
#     raw_result = db.run(generated_query)
#     # raw_result = db.run("""SELECT * FROM "vi_spo" WHERE profile_spec_name LIKE '%Прикладная информатика%' LIMIT 5;""")
#     print(raw_result)

#     import psycopg2
#     from sqlalchemy import create_engine, text
#     import pandas as pd
    
#     # Получаем соединение из объекта db
#     engine = db._engine
#     with engine.connect() as conn:
#         result_proxy = conn.execute(text(generated_query))
#         column_names = list(result_proxy.keys())
#         rows = result_proxy.fetchall()

#         result_data = []
#         for row in rows:
#             result_data.append(dict(zip(column_names, row)))
    
#     retrieved_context = []
#     for i, row_data in enumerate(result_data):
#         # Преобразуем строку в читаемый текст
#         row_text = ", ".join([f"{key}: {value}" for key, value in row_data.items()])
#         retrieved_context.append({
#             "doc_id": f"sql_{uuid.uuid4()}",
#             "text": row_text
#         })
    
#     set_retrieved_context(retrieved_context)

#     # --- Формирование ответа ---
#     format_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEXT)
#     format_chain = format_prompt | llm | StrOutputParser()

#     final_answer = format_chain.invoke({
#         'query': query,
#         'raw_result': retrieved_context
#     })

#     return final_answer