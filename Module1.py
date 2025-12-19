import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient

# ----------------------------
# Load .env locally if it exists
# ----------------------------
from dotenv import load_dotenv

# Only load .env if environment variables are not already set
from dotenv import load_dotenv
load_dotenv(override=False)

# ----------------------------
# Get required environment variables
# ----------------------------

tavily_key = os.environ.get("TAVILY_API_KEY")

# Set the environment variable explicitly for OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError("OPENAI_API_KEY is missing or empty in environment/secrets!")

if not tavily_key or tavily_key.strip() == "":
    raise ValueError("TAVILY_API_KEY is missing or empty in environment/secrets!")

# Set it (helps some libraries)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip()

print(f"OPENAI_API_KEY loaded: {bool(OPENAI_API_KEY)} (length: {len(OPENAI_API_KEY)})")
print(f"TAVILY_API_KEY loaded: {bool(tavily_key)} (length: {len(tavily_key)})")

# ----------------------------
# Initialize clients
# ----------------------------
tavily = TavilyClient(api_key=tavily_key)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY  # ← Use 'api_key', not 'openai_api_key' (correct parameter name)
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
