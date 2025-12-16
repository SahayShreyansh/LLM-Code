import streamlit as st
from click import prompt
from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


from langchain_core.globals import set_debug
import os
set_debug(True)

openai_api_key = os.getenv("OPEN_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

llm =ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

prompt_template =PromptTemplate(
        input_variables= ["city","month","language","budget"],
        template = """Welcome to the {city} travel guide !
        If you are visiting in {month},here's what you can do :
        1.Must-visit attractions.
        2.Local cuisine you must try.
        3.Useful phrases in {language}.
        4.Tips for travelling on a {budget}.
        """
)

st.title("Travel Guide")
city = st.text_input("City")
month = st.text_input("Month")
language = st.text_input("Language")
budget = st.text_input("Budget")

chain = prompt_template | llm

if city and month and language and budget:
    response = chain.invoke({
        "city": city,
        "month": month,
        "language": language,
        "budget": budget

    }
    )
    st.write(response.content)



