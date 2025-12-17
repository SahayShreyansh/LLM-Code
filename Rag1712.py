import os
import streamlit as st
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import *
from langchain_core.prompts import *
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# 1. Configuration & API Key
os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Chat with your document")

# Use the standard naming convention
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")


# 2. Cached Resource Function
@st.cache_resource
def get_retriever():
    document = TextLoader("product-data.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_documents(document)

    if os.getenv("STREAMLIT_CLOUD") == "true":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
    )
    # This stays in memory and won't re-run unless the script changes
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store.as_retriever()


# Initialize retriever
retriever = get_retriever()

# 3. Chain Setup
if os.getenv("STREAMLIT_CLOUD") == "true":
    # Use OpenAI when running on Streamlit Cloud
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
else:
    # Use Ollama when running locally
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for answering questions using the provided context. If the answer is not in the context, say you don't know. Answer in at most three concise sentences.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for answering questions using the provided context. If the answer is not in the context, say you don't know. Answer in at most three concise sentences.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 4. Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask a question about the document")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    answer = response["answer"]
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
