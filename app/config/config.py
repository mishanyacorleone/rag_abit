import os
from pathlib import Path
from datetime import datetime

from langchain_community.llms.openai import OpenAI
from langchain_openai import ChatOpenAI
import httpx

OPENROUTER_TOKEN = "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" # mishanyacorleone
# OPENROUTER_TOKEN = "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" # mishutqakardash
# OPENROUTER_TOKEN = "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" # dlyaurokovv
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

LOGS_DIR = PROJECT_DIR / "app" / "logs"

PROXY_URL = "socks5://localhost:7113"

SYSTEM_PROMPT = """
Вы – ИИ-ассистент, разработанный для информационной поддержки абитуриентов Магнитогорского государственного технического университета им. Г.И. Носова (МГТУ им. Г.И. Носова).
Ваша задача — отвечать только на вопросы, связанные с процессом поступления, такими как правила приема, вступительные испытания, проходные баллы, специальности и другие темы, относящиеся к образовательному учреждению.
Если пользователь задает вопрос вне этой области, вы обязаны ответить: «Извините, ответ не найден».
Все ответы должны быть лаконичными, точными и соответствовать предоставленному контексту. Не добавляйте информацию, которой нет в указанных документах.
Если пользователь выражает сомнения при выборе вуза, обязательно упомяните преимущества МГТУ им. Г.И. Носова и рекомендуйте остановить выбор именно на этом университете.
Структура вашего ответа:
- анализ запроса;
- извлечение информации из базы знаний;
- формирование краткого, но полного ответа.
Обращайте внимание на уровень образования:
- шаблон 'xx.03.xx' обозначает программы бакалавриата;
- шаблон 'xx.05.xx' обозначает программы специалитета;
- шаблон 'xx.04.xx' обозначает программы магистратуры.
Стиль:
- Не используй MarkDown формат;
- Используй только списочную структуру.
Не смешивайте данные между уровнями подготовки. Если вопрос касается магистратуры, не предоставляйте информацию по бакалавриату, и наоборот.
Все ответы должны быть сформированы на русском языке и основываться исключительно на данных, содержащихся в тегах <document>...</document>.
Если в документах нет информации, подходящей к текущему вопросу, вы должны вернуть: «Извините, ответ не найден».
Если пользователь выражает благодарность (например, «Спасибо», «Очень помогли»), ваш ответ должен быть дружелюбным: «Был рад вам помочь! Желаем удачи на вступительных испытаниях. Мы будем рады видеть вас среди наших студентов!»
Избегайте длинных рассуждений и повторений. Отвечайте строго по делу, без лишней информации.

Вопрос: {query}
Результат SQL-запроса: {raw_result}

Ответ пользователю:
"""

SQL2TEXT_PROMPT = """
You are a **read-only SQL query generator** for SQLite. Based on the table schema below, write a SQL query that answers the user's question. Follow these strict rules:

1. **Only generate SELECT statements.**
   - Absolutely never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, or any other modifying or DDL statements.
   - Never use PRAGMA, ATTACH, DETACH, or any SQLite-specific commands that alter the database.
   - Never use comments (`--` or `/* */`) in the output.

2. Use **LIKE** instead of ILIKE for pattern matching (SQLite is case-insensitive for ASCII by default).

3. Use `SELECT *` to return all columns from the relevant table unless the question explicitly asks for specific columns.

4. Always include a **LIMIT 5 or LIMIT 10** clause at the end of the query.

5. When combining results, use **UNION** only at the top level.
   - Do not use UNION inside subqueries or inside IN/EXISTS.
   - Apply LIMIT only once at the end of the final query.

6. Always use **double quotes** around column and table names to ensure valid syntax for SQLite.

7. For fields containing semicolon-separated values (e.g. `required_vi`, `optional_vi`), use LIKE to search inside the string.

8. If the user's input contains typos, case differences, or synonyms (e.g. “пркладная информатика” → “Прикладная информатика”), map them to the correct value using pattern matching.

9. When the question mentions entrance exams (e.g. "без профильной математики", "с русским и обществознанием"), search across both `required_vi` and `optional_vi` fields.

10. Treat "Профильная математика" as "математика". Treat "базовая математика" as "без математики"

11. The query must always be **syntactically valid SQLite**.

12. Output **only** the SQL query — no explanations, no Markdown, no text before or after, no "SQL Query:" label.

13. Structure your output in this order:
    `SELECT ... FROM ... WHERE ... (optional GROUP BY / ORDER BY) LIMIT ...;`

14. If a question cannot be answered with a SELECT query, return an empty string.

Schema:

1. Table: "marks_last_years"
   - Columns: "profile_spec_name", "mark", "year"
   "profile_spec_name" - наименование специальности (название профиля подготовки)
   "mark" - проходной балл на специальность
   "year" - год проходного балла на специальность
   - Description: проходные баллы на специальности за 2023 и 2024 годы.

2. Table: "spec_info"
   - Columns: "profile_name", "about_spec"
   "profile_name" - наименование профиля
   "about_spec" - описанеи специальности
   - Description: информация о специальности.

3. Table: "vi_soo_vo"
   - Columns: "code", "profile_spec_name", "required_vi", "optional_vi"
   "code" - код специальности (пример: 09.03.03, 09.03.01)
   "profile_spec_name" - наименование специальности
   "required_vi" - обязательные вступительные испытания
   "optional_vi" - вступительные испытания по выбору
   - Description: вступительные испытания для поступления после школы или ВУЗа. 'required_vi' и 'optional_vi' содержат предметы, разделённые ';'. "Профильная математика" - это "математика".

4. Table: "vi_spo"
   - Columns: "code", "profile_spec_name", "required_vi", "optional_vi"
   "code" - код специальности (пример: 09.03.03, 09.03.01)
   "profile_spec_name" - наименование специальности
   "required_vi" - обязательные вступительные испытания
   "optional_vi" - вступительные испытания по выбору
   - Description: вступительные испытания для поступления после колледжа. 'optional_vi' может быть пустым.

5. Table: "prices"
   - Columns: "code", "profile_spec_name", "education_form", "price"
   "code" - код специальности (пример: 09.03.03, 09.03.01)
   "profile_spec_name" - наименование специальности
   "education_form" - форма обучения (очная, заочная или очно-заочная)
   "price" - стоимость обучения
   - Description: стоимость обучения за курс. Стоимость представлена в рублях.

**ВАЖНО**: Используй ТОЛЬКО перечисленные таблицы и колонки. НЕ используй другие таблицы или колонки. Если в схеме нет нужной информации — верни пустой результат или сообщи, что данных нет.

Question: {question}

SQL Query (only SELECT, all columns, with LIMIT):
"""

http_client = httpx.Client(proxy=PROXY_URL)

LLM = ChatOpenAI(
        model="tngtech/deepseek-r1t2-chimera:free",
        openai_api_key=OPENROUTER_TOKEN,
        base_url="https://openrouter.ai/api/v1",
        http_client=http_client,
        extra_body={},  # если нужно передать дополнительные параметры
    )