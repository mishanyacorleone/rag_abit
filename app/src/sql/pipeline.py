import re
import json
import logging
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import text

from app.config.config import SQL2TEXT_PROMPT, TABLE_SELECTOR_PROMPT, PROJECT_DIR

logger = logging.getLogger(__name__)

TABLE_DESCRIPTIONS_PATH = PROJECT_DIR / "app" / "config" / "prompts" / "table_descriptions.json"

_table_descriptions = None


def _load_table_descriptions() -> dict:
    global _table_descriptions
    if _table_descriptions is None:
        with open(TABLE_DESCRIPTIONS_PATH, "r", encoding="utf-8") as file:
            _table_descriptions = json.load(file)
    return _table_descriptions


def select_tables_with_llm(query: str, llm) -> list:
    """
    LLM выбирает релевантные таблицы
    """
    descriptions = _load_table_descriptions()

    table_list = []
    for name, info in descriptions.items():
        table_list.append(f"- {name}: {info['description']}")
    
    table_list_str = "\n".join(table_list)

    prompt = ChatPromptTemplate.from_template(TABLE_SELECTOR_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "table_list": table_list_str,
        "question": query
    })

    all_tables = list(descriptions.keys())
    selected = []
    for table in all_tables:
        if table in result.lower():
            selected.append(table)
    
    if not selected:
        selected = ["spec_info", "marks_last_years"]
        logger.warning(f"LLM не выбрала таблицы, использую дефолтные: {selected}")
    
    logger.info(f"LLM выбрал таблицы: {selected}")
    return selected


def build_sql_prompt(selected_tables: list) -> str:
    """
    Собирает промпт из описаний только нужных таблиц
    """
    descriptions = _load_table_descriptions()
    header = """Ты — генератор SQL-запросов только для чтения данных из PostgreSQL. СТРОГО СОБЛЮДАЙ следующие правила и ШАБЛОНЫ для создания вопросов:

1. Генерируй ТОЛЬКО операторы SELECT. Запрещено использовать INSERT, UPDATE, DELETE, DROP, ALTER, CREATE или любые другие команды, изменяющие данные или структуру базы.
2. Все текстовые данные в базе хранятся в нижнем регистре. Не используй функции lower() или upper().
3. Всегда извлекай не более 10 строк, добавляя LIMIT 10 (или LIMIT 5 для вспомогательных таблиц).
4. Формируй SQL-запрос в одну строку без переносов.
5. Не используй markdown-разметку, включая sql. Выводи только чистый SQL-запрос.
6. Запомни: "Профильная математика" в таблице обозначена, как "математика". Есть 2 уровня математики: профильная и базовая. Если человек говорит, что он сдавал базовую математику, то значит, что ты должен искать там, где нет "математика".
7. НЕ добавляй фильтры, которых НЕТ в вопросе пользователя. Если пользователь НЕ указал форму обучения, уровень образования или год — НЕ фильтруй по ним. Покажи ВСЕ варианты.
8. Если пользователь говорит "сдавал ЕГЭ", "сдавал русский/математику/физику" или перечисляет предметы ЕГЭ — это ВСЕГДА выпускник школы (11 класс). Используй таблицу "vi_soo_vo".
9. Если в вопросе есть "самый дешёвый", "самый дорогой", "минимальный", "максимальный" — ОБЯЗАТЕЛЬНО добавляй ORDER BY с соответствующим направлением (ASC для минимальных, DESC для максимальных).
10. При использовании JOIN ВСЕГДА указывай имя таблицы перед колонкой: "таблица"."колонка". Не используй колонки без указания таблицы.
11. Всегда используй оператор LIKE для того, чтобы находить подстроку в строке по формату LIKE %ключевое_слово% 

Описания таблиц:

"""
    table_prompts = []
    for table in selected_tables:
        if table in descriptions:
            table_prompts.append(descriptions[table]["prompt"])
    
    join_rules = ""
    if len(selected_tables) > 1:
        join_rules = """ Если вопрос пользователя требует одновременно информации из нескольких таблиц, сформируй единый SQL-запрос с использованием JOIN. Используй только INNER JOIN, так как требуется точное совпадение по коду специальности.

## Основные правила объединения:
- Все таблицы содержат колонку "code" — используй её как ключ для соединения.
- Пример: пользователь спрашивает «Куда можно поступить с русским, обществом и историей, и сколько это стоит?»
- Следовательно, нужно объединить "vi_soo_vo" и "prices" по "code".
- Всегда возвращай SELECT * из всех участвующих таблиц, даже если колонки дублируются.
- Ограничение LIMIT 10 применяется один раз в конце.
Пример корректного объединённого запроса:
SELECT * FROM "vi_soo_vo" INNER JOIN "prices" ON "vi_soo_vo"."code" = "prices"."code" WHERE 'русский язык' = ANY("vi_soo_vo"."required_vi") AND ('обществознание' = ANY("vi_soo_vo"."required_vi") OR 'обществознание' = ANY("vi_soo_vo"."optional_vi_ege")) AND ('история' = ANY("vi_soo_vo"."required_vi") OR 'история' = ANY("vi_soo_vo"."optional_vi_ege")) AND NOT ('математика' = ANY("vi_soo_vo"."required_vi")) LIMIT 10;
## Другие допустимые комбинации:
- "vi_soo_vo" + "prices" — экзамены после школы и стоимость
- "vi_soo_vo" + "marks_last_years" — экзамены после школы и проходные баллы
- "vi_soo_vo" + "prices" + "marks_last_years" — полная информация для абитуриента после школыы
- "marks_last_years" + "prices" — проходные баллы и стоимость
- "spec_info" + "prices" — информация о специальности и стоимость
Если в запросе упоминается СПО (колледж) — объединяй "vi_spo" с другими таблицами аналогичным образом.
Всегда проверяй, что каждая таблица необходима для ответа. Не добавляй JOIN без явной потребности в данных из второй таблицы.

## Ключевые операторы PostgreSQL:
- "@>" — содержит (например, arr @> ARRAY['x'] означает, что массив содержит 'x')
- "&&" — пересечение (есть общие элементы)
- "LIKE" — для поиска подстроки в текстовых полях
- Массивы записываются как ARRAY['a', 'b'] или {{'a','b'}} — оба варианта допустимы, но предпочтительнее ARRAY[...]
"""
    footer = """## Финальные инструкции:
- Всегда используй синтаксис PostgreSQL.
- Не предполагай структуру таблиц за пределами приведённой.
- Если на вопрос невозможно ответить, верни пустую строку.
- Не добавляй никаких пояснений, только SQL-запрос.

Вопрос: {question}
SQL-запрос (только SELECT, все колонки, с LIMIT):
"""
    return header + "\n\n".join(table_prompts) + join_rules + footer


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
    generated_query = None

    try:
        relevant_tables = select_tables_with_llm(query, llm)

        prompt_text = build_sql_prompt(relevant_tables)
        prompt = ChatPromptTemplate.from_template(prompt_text)

        sql_chain = prompt | llm | StrOutputParser()

        generated_query_raw = sql_chain.invoke({
            "question": query
        })

        generated_query = extract_sql_query(generated_query_raw)
        logger.info(f"Сгенерированный SQL: {generated_query}")

        engine = db._engine
        with engine.connect() as conn:
            result_proxy = conn.execute(text(generated_query))
            column_names = list(result_proxy.keys())
            rows = result_proxy.fetchall()

            data = [dict(zip(column_names, row)) for row in rows]

        logger.info(f"SQL вернул: {len(data)} строк")
        return {"success": True, "query": generated_query, "data": data, "error": None}
    
    except Exception as e:
        logger.error(f"Ошибка SQL: {e}")
        return {"success": False, "query": generated_query, "data": [], "error": e}
    

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