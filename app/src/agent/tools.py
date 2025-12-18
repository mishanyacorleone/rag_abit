from langchain.tools import Tool
from app.src.sql.pipeline import sql_search
from app.src.rag.search import retrieve_docs
from app.src.rag.generate import generate_answer


def get_sql_tool(db, llm):
    """
    Возвращает инструмент для SQL-поиска.
    """
    def sql_func(query: str) -> str:
        return sql_search(query, db, llm)

    return Tool(
        name="SQLSearch",
        description="Поиск по структурированной информации: проходные баллы, специальности, вступительные испытания и т.д.",
        func=sql_func
    )


def get_vector_tool(retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents):
    """
    Возвращает инструмент для векторного поиска.
    """
    def vector_func(query: str) -> str:
        retrieved_docs = retrieve_docs(query, retriever_model, reranker_model,
                                       reranker_tokenizer, faiss_index, documents)
        docs = "\n".join([x["candidate"] for x in retrieved_docs])
        answer = generate_answer(query, docs)
        return answer

    return Tool(
        name="VectorSearch",
        description="Поиск по документам (правила, описания и т.д.). Используй, если вопрос касается правил приема, документов, процессов и т.п.",
        func=vector_func
    )