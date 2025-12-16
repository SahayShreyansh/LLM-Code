

import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI



import os



openai_api_key = os.getenv("OPEN_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

llm =ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

title_prompt = PromptTemplate(
    input_variables=["topic"],
        template="""
             You are an experienced speech writer .You need to craft an impactful title for a speech on the following topic :{topic}
             Answer exactly with one title
        """
)
speech_chain = (
    PromptTemplate(
        input_variables=["title"],
        template="Write a powerful 350-word speech for the title: {title}"
    )
    | llm
    | StrOutputParser()
)

First_chain = title_chain = title_prompt | llm | StrOutputParser()

speech_chain = (
    PromptTemplate(
        input_variables=["title"],
        template="Write a powerful 350-word speech for the title: {title}"
    )
    | llm
    | StrOutputParser()
)

final_chain = First_chain | (lambda title: {"title": title}) | speech_chain


st.title("Blog Post")
topic = st.text_input("Please enter your topic")

if topic :
    response = final_chain.invoke({"topic" : topic})
    st.write(response)


