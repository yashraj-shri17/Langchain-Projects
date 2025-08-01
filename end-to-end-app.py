import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

# =============================
# Load environment variables
# =============================
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

# =============================
# Streamlit UI Setup
# =============================
st.set_page_config(page_title="Conversational RAG with PDFs", page_icon="📚", layout="wide")

st.title("📚 Conversational RAG with PDF Uploads & Chat History")
st.write("Upload your PDF files and chat with their content using **Groq's Gemma2-9b-It** model.")

# =============================
# Sidebar - API Key & Session
# =============================
with st.sidebar:
    st.subheader("🔑 API & Settings")
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    session_id = st.text_input("Session ID", value="default_session")
    uploaded_files = st.file_uploader("📄 Upload PDF files", type="pdf", accept_multiple_files=True)

# =============================
# Initialize variables
# =============================
retriever = None
conversational_rag_chain = None

# =============================
# Function: Session History Store
# =============================
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# =============================
# If API key provided
# =============================
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # =============================
    # If PDFs uploaded, process them
    # =============================
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create vectorstore & retriever
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

    # =============================
    # Create History-Aware Retriever & RAG Chain
    # =============================
    if retriever:
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriver = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriver, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    # =============================
    # Chat Interface
    # =============================
    user_input = st.chat_input("💬 Ask a question about your PDFs")

    if user_input:
        if conversational_rag_chain is None:
            st.warning("⚠️ Please upload at least one PDF before asking a question.")
        else:
            session_history = get_session_history(session_id)

            # Invoke the RAG chain
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # Display past chat messages first
            for msg in session_history.messages:
                if msg.type == "human":
                    st.chat_message("user").write(msg.content)
                else:
                    st.chat_message("assistant").write(msg.content)

            # Show current interaction
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write(response["answer"])

else:
    st.warning("Please enter your Groq API Key in the sidebar to start chatting.")
