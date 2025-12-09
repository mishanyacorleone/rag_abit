from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from app.config.config import OPENROUTER_TOKEN, PROXY_URL, OPENROUTER_MODEL
import httpx

from app.src.agent.tools import get_sql_tool, get_vector_tool

def initialize_agent_with_tools(retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents, db):
    """
    Инициализирует агента с SQL и Vector инструментами.
    """
    http_client = httpx.Client(proxy=PROXY_URL)

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_TOKEN,
        base_url="https://openrouter.ai/api/v1",
        http_client=http_client,
        extra_body={},
    )

    tools = [
        get_sql_tool(db, llm),
        get_vector_tool(retriever_model, reranker_model, reranker_tokenizer, faiss_index, documents)
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent