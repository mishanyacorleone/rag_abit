from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from app.config.config import SQL2TEXT_PROMPT, LLM
import httpx

db = SQLDatabase.from_uri("sqlite:///../../../data/university.db")


def load_sql_components(query: str):
    prompt = ChatPromptTemplate.from_template(SQL2TEXT_PROMPT)

    def get_schema(_):
        return db.get_table_info()

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | LLM
        | StrOutputParser()
    )

    generated_query = sql_response.invoke({"question": query})
    result = db.run(generated_query)

    return sql_response


# Ответ на вопрос;
# [1][2][3]
# 1. https://magtu.ru - текст ответа
# 2. https://abit.magtu.ru - текст ответа