import os
from platform import system

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import *
from langchain_community.chat_message_histories import *
from langchain_core.globals import set_debug
from langchain_core.runnables.history import RunnableWithMessageHistory


set_debug(True)

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not set")
llm = ChatOpenAI(model ="gpt-4o",api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(
              [
                  ("system","You are a Agile coach.Answer any questions"
                   "related to the agile process"),
                  MessagesPlaceholder(variable_name= "chat_history"),
                  ("human" ,"{input}")
              ]

)
chain = prompt_template | llm

history_for_chain = StreamlitChatMessageHistory()

chain_with_history =RunnableWithMessageHistory(chain,

                             lambda session_id : history_for_chain,
                                               input_messages_key="input",
                                               history_messages_key="chat_history"
                                    )
st.title("Agile Guide")
input = st.text_input("Enter your Question")

if input is not None:
    response = chain_with_history.invoke(
        {"input": input},
        config={"configurable": {"session_id": "abc123"}}
    )
    st.write(response.content)
    st.write(history_for_chain)