from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from app.config.config import SQL2TEXT_PROMPT, SYSTEM_PROMPT
from datetime import datetime
from typing import Optional
import re


def sql_search(query: str, db: SQLDatabase, llm) -> str:
    """
    Генерирует и выполняет SQL-запрос к базе данных.

    :param query: Вопрос пользователя.
    :param db: Объект SQLDatabase.
    :param llm: Объект LLM (например, ChatOpenAI).
    :return: Результат выполнения SQL-запроса.
    """
    prompt = ChatPromptTemplate.from_template(SQL2TEXT_PROMPT)

    def get_schema(_):
        return db.get_table_info()

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Генерация запроса ---
    generated_query_raw = sql_response.invoke({"question": query})
    generated_query = extract_sql_query(generated_query_raw)

    # --- Выполнение запроса ---
    print(generated_query)
    # raw_result = db.run(generated_query)
    raw_result = db.run(generated_query)
    # raw_result = db.run("""SELECT * FROM "vi_spo" WHERE profile_spec_name LIKE '%Прикладная информатика%' LIMIT 5;""")
    print(raw_result)
    # --- Формирование ответа ---
    format_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    format_chain = format_prompt | llm | StrOutputParser()

    final_answer = format_chain.invoke({
        'query': query,
        'raw_result': raw_result
    })

    return final_answer


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