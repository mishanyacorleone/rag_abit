# from langchain.agents import initialize_agent, AgentType, create_structured_chat_agent, AgentExecutor
# from langchain.tools import Tool
# from langchain_openai import ChatOpenAI
# from app.config.config import CUSTOM_AGENT_PROMPT, get_settings
# import httpx

# from app.src.agent.tools import get_sql_tool, get_vector_tool

# def initialize_agent_with_tools(retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents, db):
#     """
#     Инициализирует агента с SQL и Vector инструментами.
#     """

#     settings = get_settings()

#     llm = ChatOpenAI(
#         base_url=settings.vllm.base_url,
#         api_key=settings.vllm.api_key,  # vLLM не требует ключа
#         model=settings.vllm.model_name
#     )

#     tools = [
#         get_sql_tool(db, llm),
#         get_vector_tool(retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents)
#     ]

#     agent = create_structured_chat_agent(
#         llm=llm,
#         tools=tools,
#         prompt=CUSTOM_AGENT_PROMPT,

#     )

#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         verbose=True,
#         handle_parsing_errors=True,
#         max_iterations=5,
#         early_stopping_method="generate",
#     )

#     # agent = initialize_agent(
#     #     tools,
#     #     llm,
#     #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     #     verbose=True,
#     #     handle_parsing_errors=True
#     # )

#     return agent_executor

















import logging
from langchain.agents import create_structured_chat_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config.config import CUSTOM_AGENT_PROMPT, get_settings, SYSTEM_PROMPT_TEXT
from app.src.agent.tools import get_sql_tool, get_vector_tool

logger = logging.getLogger(__name__)


def initialize_agent_with_tools(
        retriever_model, reranker_model, reranker_tokenizer,
        qdrant_client, collection_name, db
):
    """
    Инициализирует агента с SQL и Vector инструментами
    """

    settings = get_settings()

    llm = ChatOpenAI(
        base_url=settings.vllm.base_url,
        api_key=settings.vllm.api_key,
        model=settings.vllm.model_name,
        temperature=0
    )

    tools = [
        get_sql_tool(db, llm),
        get_vector_tool(
            retriever_model, reranker_model, reranker_tokenizer,
            qdrant_client, collection_name
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEXT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        # early_stopping_method='generate',
        return_intermediate_steps=True
    )

    return agent_executor
