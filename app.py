import os
import streamlit as st
import torch
from typing import List, Dict, Any

# Hugging Face and LangChain imports
from langchain_community.llms import HuggingFaceEndpoint
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Additional libraries
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API Token (consider using environment variable securely)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")

class PDFAssistant:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        """
        Initialize the PDF assistant with a specified language model
        """
        # Initialize device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load LLM
        self.llm = self._load_llm(model_name)
        
        # Initialize embeddings
        self.embeddings = self._load_embeddings()

    def _load_llm(self, model_name):
        """Load the language model"""
        try:
            return HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=HF_TOKEN,
                task="text-generation",
                max_new_tokens=512
            )
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def _load_embeddings(self):
        """Load embeddings for vector store"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

    def process_pdf(self, file_bytes):
        """
        Process uploaded PDF file
        
        Args:
            file_bytes (bytes): PDF file content
        
        Returns:
            list: Split document chunks
        """
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(file_bytes)
        
        loader = PyMuPDFLoader(temp_pdf_path)
        docs = loader.load()
        os.remove(temp_pdf_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        return text_splitter.split_documents(docs)

    def create_vector_db(self, chunks):
        """
        Create vector database from document chunks
        
        Args:
            chunks (list): Document chunks
        
        Returns:
            FAISS: Vector database
        """
        return FAISS.from_documents(chunks, self.embeddings)

def main():
    # Streamlit app configuration
    st.set_page_config(page_title="PDF Assistant", layout="wide")
    st.title("🤖 PDF Assistant")

    # Sidebar for file upload
    st.sidebar.title("Document Settings")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    
    # Reset session state when a new file is uploaded
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None

    # Check if a new file is uploaded
    if uploaded_file and uploaded_file != st.session_state.last_uploaded_file:
        # Clear existing session state
        for key in list(st.session_state.keys()):
            if key not in ['last_uploaded_file', 'pdf_uploader']:
                del st.session_state[key]
        
        # Update last uploaded file
        st.session_state.last_uploaded_file = uploaded_file

    # Initialize assistant
    if 'assistant' not in st.session_state:
        try:
            st.session_state.assistant = PDFAssistant()
        except Exception as e:
            st.error(f"Failed to initialize assistant: {e}")
            st.stop()
    
    # Document processing
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            chunks = st.session_state.assistant.process_pdf(uploaded_file.getvalue())
            vector_db = st.session_state.assistant.create_vector_db(chunks)
        
        st.sidebar.success("PDF Processed Successfully!")

        # Chat with Document interface
        # Conversational Retrieval Chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.assistant.llm,
            retriever=vector_db.as_retriever(),
            memory=memory
        )

        # Chat interface
        st.subheader("💬 Document Chat")
        for message in st.session_state.get('chat_history', []):
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        
        user_query = st.chat_input("Ask a question about the document...")
        if user_query:
            st.session_state.chat_history = st.session_state.get('chat_history', [])
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.spinner("Thinking..."):
                response = conversation_chain({"question": user_query})["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    else:
        st.info("Upload a PDF to start chatting with your document!")

if __name__ == "__main__":
    main()