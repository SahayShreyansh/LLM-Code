import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_debug
set_debug(True)
import  os


# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Streaming LLM Chat",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ Ollama Streaming Chat")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        ["gemma:2b", "llama3", "mistral"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# -----------------------------
# Streaming Callback
# -----------------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# -----------------------------
# Cache LLM
# -----------------------------
@st.cache_resource
def load_llm(model: str, temperature: float):
    return ChatOllama(
        model=model,
        temperature=temperature,
        streaming=True,
    )

llm = load_llm(model_name, temperature)

# -----------------------------
# Input
# -----------------------------
question = st.text_input(
    "Ask anything",
    placeholder="Type your question...",
)

# -----------------------------
# Streaming Response
# -----------------------------
if question:
    response_container = st.empty()
    handler = StreamHandler(response_container)

    try:
        with st.spinner("Streaming response..."):
            llm.invoke(
                [HumanMessage(content=question)],
                config={"callbacks": [handler]},
            )

    except Exception as e:
        st.error("Failed to generate response")
        st.exception(e)
else:
    st.info("👆 Enter a question to start")
