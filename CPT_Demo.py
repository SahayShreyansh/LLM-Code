import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import *
from langchain_core.globals import set_debug



set_debug(True)

OPENAI_API_KEY = os.environ.get("OPEN_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not set")

llm =ChatOpenAI(model ="gpt-4o", api_key=OPENAI_API_KEY)

prompt_template =ChatPromptTemplate.from_messages(
    [
        ("system" ,"You are a Agile Coach.Answer any question related to the agile process")
        ,("human" ,"{input}")
    ]
)
st.title("Agile Guide")
input = st.text_input("Enter the question : ")

chain = prompt_template |llm
if st.button("Generate Question"):
    if input :
          response = chain.invoke({"input":input})
          st.write(response.content)
