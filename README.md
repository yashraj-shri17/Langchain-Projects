Here is a `README.md` file for your **Conversational RAG with PDF Uploads** app based on the uploaded `end-to-end-app.py`:

---

````markdown
# 📚 Conversational RAG with PDF Uploads & Chat History

This Streamlit application allows you to chat with the content of your PDF documents using a **Conversational Retrieval-Augmented Generation (RAG)** pipeline. It leverages **Groq's Gemma2-9b-It** model and supports persistent session-based chat history.

---

## 🚀 Features

- 🔐 Input your Groq API Key securely
- 📄 Upload one or multiple PDF files
- 💬 Ask questions about the contents of your PDFs
- 🧠 Uses HuggingFace Embeddings and Chroma Vectorstore
- 💾 Session-specific chat history using Streamlit state
- 🔁 Automatically reformulates context-aware questions
- 📚 Powered by LangChain, Streamlit, and Groq

---

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install streamlit langchain langchain-community langchain-core langchain-chroma \
            langchain-groq langchain-huggingface python-dotenv
````

You also need `chromadb`, `PyMuPDF`, and other dependencies for PDF and vectorstore handling.

---

## 🔧 Setup Instructions

1. **Clone the repository** (or place the script in your working directory).

2. **Create a `.env` file** in the root folder and add:

```
HF_TOKEN=your_huggingface_token_here
```

3. **Run the app**:

```bash
streamlit run end-to-end-app.py
```

---

## 🛠 How It Works

* **PDF Processing**: PDFs are loaded and split into chunks using LangChain's `RecursiveCharacterTextSplitter`.
* **Vector Store**: Chunks are embedded using HuggingFace embeddings and stored in ChromaDB.
* **RAG Pipeline**:

  * A history-aware retriever reformulates questions using chat history.
  * Retrieved documents are passed to the QA chain.
* **Frontend**: Powered by Streamlit with sidebar inputs and chat-style interface.

---

## 🔑 API Key Usage

To use the Groq LLM model (`Gemma2-9b-It`), you must enter your **Groq API key** in the sidebar. Without it, the application won't process chats.

---

## 📁 File Upload and Session History

* Multiple PDF uploads are supported.
* A `Session ID` lets you maintain separate chat sessions.

---

## 🧠 Models Used

* **LLM**: Groq's `Gemma2-9b-It`
* **Embeddings**: HuggingFace's `all-MiniLM-L6-v2`

---

## 📸 Screenshot

![App Screenshot](https://placehold.co/800x400?text=Upload+PDFs+and+Chat+in+Real-time)

---

## 🤝 Credits

Developed using:

* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [Groq](https://groq.com/)
* [ChromaDB](https://www.trychroma.com/)
* [HuggingFace](https://huggingface.co/)

---

## 📃 License

MIT License. Use it for educational or research purposes freely.

```

---
