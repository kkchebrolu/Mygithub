# multi_file_qa_chatbot.py
# Run: streamlit run multi_file_qa_chatbot.py

import os
import glob
import streamlit as st
import torch

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen3-1.7B"
DB_DIR = "./multi_file_chroma_db"

st.set_page_config(
    page_title="Multi-file Q&A Chatbot (PDF / Excel / CSV)",
    page_icon="📂",
    layout="wide"
)
st.title("📂 Folder Q&A Chatbot (PDF, Excel, CSV)")
st.caption("Point to a folder, index all documents, and chat with Qwen3-1.7B over them.")

# =========================
# Model & Embeddings
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        #device_map="auto",
        trust_remote_code=True
    )
    
    return tokenizer, model

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# Document Loading
# =========================
def load_all_documents(folder_path: str):
    docs = []

    pdf_files = glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    xlsx_files = glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)
    xls_files = glob.glob(os.path.join(folder_path, "**", "*.xls"), recursive=True)

    # PDFs
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source_file"] = file_path
                d.metadata["file_type"] = "pdf"
            docs.extend(file_docs)
        except Exception as e:
            st.warning(f"Could not read PDF: {file_path} | {e}")

    # CSVs
    for file_path in csv_files:
        try:
            loader = CSVLoader(file_path=file_path)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source_file"] = file_path
                d.metadata["file_type"] = "csv"
            docs.extend(file_docs)
        except Exception as e:
            st.warning(f"Could not read CSV: {file_path} | {e}")

    # Excel (xlsx + xls)
    for file_path in xlsx_files + xls_files:
        try:
            loader = UnstructuredExcelLoader(file_path)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source_file"] = file_path
                d.metadata["file_type"] = "excel"
            docs.extend(file_docs)
        except Exception as e:
            st.warning(f"Could not read Excel: {file_path} | {e}")

    return docs, pdf_files, csv_files, xlsx_files, xls_files

# =========================
# Vector Store / Retriever
# =========================
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    return retriever, len(chunks)

# =========================
# Q&A with Qwen3-1.7B
# =========================
def ask_question(tokenizer, model, retriever, question: str):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for question answering over PDF, Excel, "
                "and CSV files. Answer only from the provided context. If the answer "
                "is not in the context, say you do not know."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip(), retrieved_docs

# =========================
# Session State
# =========================
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📁 Folder Settings")
    default_path = os.getcwd()
    folder_path = st.text_input(
        "Folder path",
        value=r"C:\Users\kich0414\Desktop\AIassignments",
        help="Example: C:/Users/Name/Documents/data or /home/user/data"
    )
    

    build_btn = st.button("🔍 Build / Rebuild Index", use_container_width=True)
    clear_chat_btn = st.button("🗑️ Clear Chat", use_container_width=True)

if clear_chat_btn:
    st.session_state.chat_history = []
    st.experimental_rerun()

# =========================
# Build Index
# =========================
tokenizer, model = load_model()

if build_btn:
    if not os.path.exists(folder_path):
        st.error("❌ Folder path does not exist.")
    else:
        with st.spinner("📂 Reading all PDF, Excel, and CSV files..."):
            docs, pdfs, csvs, xlsx_files, xls_files = load_all_documents(folder_path)

        st.write(f"📄 PDF files found: {len(pdfs)}")
        st.write(f"📑 CSV files found: {len(csvs)}")
        st.write(f"📊 XLSX files found: {len(xlsx_files)}")
        st.write(f"📊 XLS files found: {len(xls_files)}")
        st.write(f"📚 Total document objects: {len(docs)}")

        if len(docs) == 0:
            st.warning("No supported files were loaded. Please check the folder.")
        else:
            with st.spinner("🧠 Building vector database..."):
                retriever, chunk_count = build_vectorstore(docs)
                st.session_state.retriever = retriever
            st.success(f"✅ Index built successfully with {chunk_count} chunks.")

# =========================
# Chat UI
# =========================
st.markdown("---")
col_chat, col_sources = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Chat")

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    question = st.chat_input("Ask a question about the folder files...")

    if question:
        if st.session_state.retriever is None:
            st.error("Please build the index first (sidebar).")
        else:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer, sources = ask_question(
                        tokenizer,
                        model,
                        st.session_state.retriever,
                        question
                    )
                    st.markdown(answer)

            st.session_state.chat_history.append((question, answer))

with col_sources:
    st.subheader("📚 Last Answer Sources")
    if "retriever" in st.session_state and st.session_state.retriever:
        st.caption("Top chunks used per question will be shown below the answer.")
    else:
        st.caption("Build the index first to see sources.")

st.markdown("---")
st.markdown(
    "**Tech:** Qwen3-1.7B· LangChain loaders for PDF/CSV/Excel · ChromaDB · sentence-transformers embeddings."
)