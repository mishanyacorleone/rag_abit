import uuid
import logging
from typing import List, Dict

from langchain.tools import Tool

from app.src.sql.pipeline import sql_query_only
from app.src.rag.search import retrieve_docs
from app.src.rag.generate import generate_answer
from app.src.agent.context_buffer import set_retrieved_context


logger = logging.getLogger(__name__)


def get_vector_tool(retriever_model, reranker_model, reranker_tokenizer,
                    qdrant_client, collection_name):
    """
    Возвращает инструмент для векторного поиска
    Возвращает найденные документы, а не готовый ответ
    """

    # def vector_func(query: str) -> str:
    #     results = retrieve_docs(
    #         query, retriever_model, reranker_model,
    #         reranker_tokenizer, qdrant_client, collection_name
    #     )

    #     if not results:
    #         return "По данному запросу документы не найдены."

    #     docs = "\n".join([x["candidate"] for x in results])
    #     answer = generate_answer(query, docs)

    #     # Контекст для RAGChecker
    #     retrieved_context = []
    #     for doc in results:
    #         retrieved_context.append({
    #             "doc_id": f"vector_{uuid.uuid4()}",
    #             "text": doc["candidate"]
    #         })
    #     set_retrieved_context(retrieved_context)

    #     return answer

    def vector_func(query: str) -> str:
        results = retrieve_docs(
            query, retriever_model, reranker_model,
            reranker_tokenizer, qdrant_client, collection_name
        )

        if not results:
            logger.warning(f"Документы не найдены")
            return "По данному запросу документы не найдены"
        
        logger.info(f"Найдено {len(results)} документов:")
        for i, doc in enumerate(results, 1):
            score = doc.get("score", 0)
            text_preview = doc["candidate"][:200]
            logger.info(f"[{i}] score={score:.4f} | {text_preview}...")

        retrieved_context = []
        result_parts = []

        for i, doc in enumerate(results, 1):
            text = doc["candidate"]
            score = doc.get("score", 0)

            retrieved_context.append({
                "doc_id": f"vector_{uuid.uuid4()}",
                "text": text
            })

            result_parts.append(
                f"[Документ {i}] (релевантность: {score:.2f})\n{text}"
            )

        set_retrieved_context(retrieved_context)

        return "\n\n".join(result_parts)
    
    return Tool(
        name="VectorSearch",
        description=(
            "Поиск по документам приемной комиссии: правила приема, " \
            "порядок подачи документов, сроки приема, льготы, общежитие. " \
            "НЕ используй для поиска специальностей, баллов, стоимости."
        ),
        func=vector_func
    )


def get_sql_tool(db, llm):
    """
    Возвращает инструмент для SQL-поиска.
    Генерирует SQL, выполняет и возвращает сырые данные.
    """

    def sql_func(query: str) -> str:
        result = sql_query_only(query, db, llm)
        
        if not result["success"]:
            return f"Ошибка SQL-запроса: {result["error"]}"
        
        retrieved_context = []
        for row in result["data"]:
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            retrieved_context.append({
                "doc_id": f"sql_{uuid.uuid4()}",
                "text": row_text
            })

        set_retrieved_context(retrieved_context)

        if not result["data"]:
            return (
                f"SQL-запрос выполнен, но данных не найдено.\n"
                f"Запрос: {result["query"]}"
            )
        
        rows_text = []
        for row in result["data"]:
            row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
            rows_text.append(row_text)
        
        return (
            f"Найдено {len(result["data"])} результатов: \n\n"
            + "\n".join(rows_text)
        )
    
    return Tool(
        name="SQLSearch",
        description=(
            "Поиск структурированной информации: специальности, направления, "
            "проходные баллы, минимальные баллы по предметам, стоимость обучения, "
            "вступительные испытания после школы, вступительные испытания после колледжа, "
            "вступительные испытания в магистратуру, дистанционные экзамены, "
            "количество бюджетных мест, куда поступить после колледжа, "
            "какие специальности доступны после СПО, соответствие специальностей СПО и ВО. "
            "Используй для ЛЮБЫХ вопросов про цифры, специальности, экзамены, "
            "баллы, стоимость, колледж, магистратуру, формы сдачи экзаменов."
        ),
        func=sql_func,
    )


# def append_rag_data_to_json(
#         query: str, 
#         response: str, 
#         retrieved_context: List[Dict[str, str]], 
#         json_file_path: str = "/mnt/mishutqa/PycharmProjects/abitBot/data/parse/ragchecker_data.json"
#     ):
#     """
#     Добавляет данные RAG запроса в JSON файл.
    
#     Args:
#         query (str): Вопрос пользователя
#         response (str): Ответ, сгенерированный RAG системой
#         retrieved_context (List[Dict[str, str]]): Контекст, извлеченный из базы знаний
#         json_file_path (str): Путь к JSON файлу для сохранения данных
#     """

#     query_id = str(uuid.uuid4())
#     entry = {
#         "query_id": query_id,
#         "query": query,
#         "gt_answer": "",  # ground truth answer пока пустой, его нужно заполнять вручную
#         "response": response,
#         "retrieved_context": retrieved_context
#     }

#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             try:
#                 data = json.load(f)
#                 if not isinstance(data, list):
#                     data = []
#             except json.JSONDecodeError:
#                 data = []
#     except FileNotFoundError:
#         data = []

#     data.append(entry)

#     with open(json_file_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

#     print(f"Запись добавлена в {json_file_path} с ID: {query_id}")



# def get_sql_tool(db, llm):
#     """
#     Возвращает инструмент для SQL-поиска.
#     """
#     def sql_func(query: str) -> str:
#         return sql_search(query, db, llm)

#     return Tool(
#         name="SQLSearch",
#         description="Поиск по структурированной информации: проходные баллы, специальности, вступительные испытания и т.д. Используй, если вопрос касается описания специальностей, проходных баллов, стоимости, вступительных испытаний, куда поступать с теми или иными экзаменами и так далее",
#         func=sql_func
#     )


# def get_vector_tool(retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents):
#     """
#     Возвращает инструмент для векторного поиска.
#     """
#     def vector_func(query: str) -> str:
#         retrieved_docs = retrieve_docs(query, retriever_model, reranker_model,
#                                        reranker_tokenizer, faiss_index, documents)
#         docs = "\n".join([x["candidate"] for x in retrieved_docs])
#         answer = generate_answer(query, docs)

#         retrieved_context = []
#         for doc in retrieved_docs:
#             retrieved_context.append(
#                 {
#                     "doc_id": f"vector_{str(uuid.uuid4())}",
#                     "text": doc["candidate"]
#                 }
#             )

#         from app.src.agent.context_buffer import set_retrieved_context
#         set_retrieved_context(retrieved_context)

#         return answer

#     return Tool(
#         name="VectorSearch",
#         description="Поиск по документам (правила приема). Используй, если вопрос касается правил приема, документов, процессов и т.п. Не используй, если касается специальностей.",
#         func=vector_func
#     )


