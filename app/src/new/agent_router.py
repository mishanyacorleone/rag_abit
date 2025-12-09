from langchain.tools import Tool
from app.config.config import LLM
from app.src.rag.search import retrieve_docs
from search_in_db import sql_search
from langchain.agents import initialize_agent, AgentType

vector_tool = Tool(
    name='VectorSearch',
    description='Поиск по документам (правила, описания и т.д.). Используй, если вопрос касается правил приема, документов, процессов и т.п.',
    func=retrieve_docs
)

sql_tool = Tool(
    name='SqlSearch',
    description='Поиск по структурированной информации: проходные баллы за прошлые года, стоимость обучения, информация о специальностях, вступительные испытания для абитуриентов, поступающих после колледжа, школы и высшего образования',
    func=sql_search
)

tools = [vector_tool, sql_tool]

agent = initialize_agent(
    tools,
    LLM,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

question = 'Какие документы нужны при поступлении?'
response = agent.run(question)
print(response)

