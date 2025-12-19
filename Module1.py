import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient

# ----------------------------
# Verify required environment variables
# ----------------------------
openai_key = os.environ.get("OPENAI_API_KEY")
print(openai_key)
tavily_key = os.environ.get("TAVILY_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY is missing!")
if not tavily_key:
    raise ValueError("TAVILY_API_KEY is missing!")

print("OPENAI_API_KEY exists and loaded.")
print("TAVILY_API_KEY exists and loaded.")

# ----------------------------
# Initialize clients
# ----------------------------
tavily = TavilyClient(api_key=tavily_key)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_key  # explicitly pass the key
)

# ----------------------------
# Define a tool for web search using Tavily
# ----------------------------
@tool
def web_search(query: str) -> Dict[str, Any]:
    "Search the web for information"
    return tavily.search(query=query)

# Example of invoking the tool standalone
result = web_search.invoke("What is the capital of Burkina Faso?")
print("Web search result:", result)

# ----------------------------
# Create an agent with the LLM and the tool
# ----------------------------
model = create_agent(model=llm, tools=[web_search])

# Invoke the agent with a question
response = model.invoke({
    "messages": [HumanMessage(content="What is the capital of Burkina Faso?")]
})

print("Agent response:", response["messages"][-1].content)
