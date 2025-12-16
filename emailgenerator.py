import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.globals import set_debug

# -----------------------------
# Debug
# -----------------------------
set_debug(True)

# -----------------------------
# OpenAI Config
# -----------------------------
openai_api_key = os.getenv("OPEN_API_KEY")
if not openai_api_key:
    raise ValueError("OPEN_API_KEY is not set")

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.7
)

# -----------------------------
# Prompt: Subject Line
# -----------------------------
product_prompt = PromptTemplate(
    input_variables=["product_name", "feature"],
    template="""
You are an experienced marketing specialist.

Create a catchy email subject line for the product:
Product: {product_name}

Key features:
{feature}

Respond with ONLY the subject line.
"""
)

# -----------------------------
# Prompt: Email Body
# -----------------------------
email_prompt = PromptTemplate(
    input_variables=["subject_line", "product_name", "target_audience"],
    template="""
Write a concise marketing email (50 words).

Product: {product_name}
Target audience: {target_audience}

Return STRICTLY valid JSON only (no markdown, no extra text):

{{
  "subject": "{subject_line}",
  "audience": "{target_audience}",
  "email": "Email body text here"
}}
"""
)

subject_chain = product_prompt | llm | StrOutputParser()

final_chain = (
    RunnablePassthrough.assign(
        subject_line=subject_chain
    )
    | email_prompt
    | llm
    | JsonOutputParser()
)

st.set_page_config(page_title="Marketing Email Generator")
st.title("Marketing Email Generator")

product_name = st.text_input("Product name")
features = st.text_input("Product features (comma separated)")
target_audience = st.text_input("Target audience")

if st.button("Generate Email"):
    if not product_name or not features or not target_audience:
        st.warning("Please fill in all fields.")
    else:

        response = final_chain.invoke({
                "product_name": product_name,
                "feature": features,
                "target_audience": target_audience
            })


        st.json(response)
