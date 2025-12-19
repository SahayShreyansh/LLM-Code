import os
from dotenv import load_dotenv  # 1. Import the loader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatOpenAI(
    model="gpt-4o-mini",
    )


@tool
def web_search(query:str) ->Dict[str, any]:
    "Search the web for information"
    return tavily.search(query=query)
web_search.invoke("What is the capital of Burkina Faso?")

model = create_agent(model=llm,tools=[web_search])

response = model.invoke({ "messages": [HumanMessage(content="What is the capital of Burkina Faso?")]})
print(response["messages"][-1].content)
