# Pdfparser

**Pdfparser** is a Streamlit-based application that enables users to upload and interact with PDF documents. It uses Google Generative AI for intelligent question-answering and FAISS for efficient similarity search.

## Features

- **Upload PDFs**: Easily upload and process multiple PDF files.
- **Text Extraction**: Extracts text along with page numbers from the uploaded PDFs.
- **Vector Search**: Utilizes FAISS to perform similarity search on document chunks.
- **AI-Powered Q&A**: Answers user questions using Google Generative AI.
- **Contextual Responses**: Ensures accurate answers or indicates when context is insufficient.

## Requirements

To run the application, make sure the following dependencies are installed:

- Python 3.9 or higher
- Streamlit
- PyPDF2
- FAISS
- LangChain
- Google Generative AI Python SDK
- Python `dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdfparser.git
   cd pdfparser

2. Download the requirements
   ```bash
   pip install -r requirements.txt

3. Set up your .env file

4. Run the Streamlit application:
```bash
streamlit run app.py
