# PDF Assistant

## Overview
PDF Assistant is a Streamlit-based web application that allows users to interact with PDF documents using natural language. The application uses Hugging Face language models and LangChain to provide an intelligent document assistant that can answer questions about uploaded PDFs.

## Features
- 📄 PDF document upload and processing
- 💬 Interactive chat interface with contextual question-answering
- 🔍 Semantic search through document content
- 🧠 Powered by Mistral-7B-Instruct language model
- 📊 Document chunking and vector embedding for efficient retrieval

## Tech Stack
- **Streamlit**: Web interface
- **LangChain**: Document processing and retrieval chains
- **Hugging Face Transformers**: Language models and embeddings
- **FAISS**: Vector database for semantic search
- **PyMuPDF**: PDF processing

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Other
- This application can be deployed to cloud platforms like Streamlit Cloud, Heroku, or AWS
- For production use, consider implementing user authentication
- The application supports any PDF document, but performance may vary based on document length and complexity
- Consider using a more powerful language model for complex documents
- Ensure your Hugging Face API token has the necessary permissions for model access

## Usage
1. Start the Streamlit app:
   **streamlit run r.py**

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the sidebar to upload a PDF document

4. Once processed, ask questions about the document in the chat interface

## How It Works
1. The application splits the uploaded PDF into manageable chunks
2. Text is converted into vector embeddings using a sentence transformer model
3. Embeddings are stored in a FAISS vector database for efficient retrieval
4. When you ask a question, the system:
   - Converts your question into the same embedding space
   - Finds the most relevant document chunks
   - Sends the relevant context along with your question to the language model
   - Returns a contextualized response based on the PDF content

## Configuration Options
- You can modify the `chunk_size` and `chunk_overlap` parameters in the `process_pdf` method to optimize for different types of documents
- Change the model by modifying the `model_name` parameter in the PDFAssistant initialization

## Requirements
- streamlit
- torch
- langchain
- langchain_community
- huggingface_hub
- sentence-transformers
- faiss-cpu (or faiss-gpu for GPU acceleration)
- pymupdf
- python-dotenv

**Link:**https://pdfassistant-1572.streamlit.app/
