import os
from langchain_openai import *

openai_api_key = os.getenv("OPEN_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

llm =ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

question = input("What is your question?")
response = llm.invoke(question)
print(response)