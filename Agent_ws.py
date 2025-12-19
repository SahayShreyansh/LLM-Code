import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

# 1. Load the keys from the .env file
load_dotenv()

# 2. Initialize the model (it will automatically look for OPENAI_API_KEY in env)
llm = ChatOpenAI(model="gpt-4o")

@tool
def web_search(query: str):
    """Search the web for information using Tavily."""
    # This automatically uses TAVILY_API_KEY from your .env

    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily.search(query=query)

# 3. Create the agent
tools = [web_search]
agent_executor = create_react_agent(llm, tools)

# 4. Run a query
question = "Who has won world cup cricket most number of times?"
try:
    response = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    # The last message in the list is the AI's final answer
    print(response['messages'][-1].content)
except Exception as e:
    print(f"Authentication Error: {e}. Check if your NEW API keys are set in .env")