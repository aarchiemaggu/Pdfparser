# Importing necessary libraries
import streamlit as st  # For building the web app interface
from PyPDF2 import PdfReader  # For reading PDF documents and extracting text
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting large chunks of text into smaller chunks
import os  # For handling environment variables and file operations
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For embedding generation using Google Generative AI
import google.generativeai as genai  # For configuring and using Google's generative AI models
from langchain.vectorstores import FAISS  # For working with FAISS vector stores for efficient search
from langchain_google_genai import ChatGoogleGenerativeAI  # For using Google Generative AI for conversation and QA
from langchain.chains.question_answering import load_qa_chain  # For creating a QA chain using a language model
from langchain.prompts import PromptTemplate  # For creating prompts to guide the model's behavior
from dotenv import load_dotenv  # For loading environment variables from a .env file
from langchain.schema import Document  # For working with Langchain's Document schema

# Load environment variables
load_dotenv()  # Load the environment variables from the .env file
google_api_key = os.getenv("GOOGLE_API_KEY")  # Get the Google API key from environment

# Check if the Google API key is provided
if not google_api_key:
    st.error("Google API key is missing. Please set it in the .env file.")  # Display an error if API key is missing
else:
    # Configure Google Generative AI with the API key
    genai.configure(api_key=google_api_key)

# Function to extract text from uploaded PDF documents
def get_pdf_text(pdf_docs):
    text_with_page_numbers = []  # List to store text along with page numbers
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Read the PDF document
        for page_num, page in enumerate(pdf_reader.pages):  # Iterate through all pages of the PDF
            text = page.extract_text()  # Extract the text from the page
            if text:
                # Append the extracted text along with its page number (starting from 1)
                text_with_page_numbers.append((text, page_num + 1))
    return text_with_page_numbers  # Return the list of text and page numbers

# Function to split the extracted text into smaller chunks
def get_text_chunks(text_with_page_numbers):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)  # Split large chunks of text
    chunks_with_page_numbers = []  # List to store text chunks along with their page numbers
    for text, page_num in text_with_page_numbers:
        chunks = text_splitter.split_text(text)  # Split the text into smaller chunks
        for chunk in chunks:
            chunks_with_page_numbers.append((chunk, page_num))  # Append chunk and its corresponding page number
    return chunks_with_page_numbers  # Return the list of chunks with page numbers

# Function to create a FAISS vector store from the text chunks
def get_vector_store(chunks_with_page_numbers):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use Google Generative AI to generate embeddings
        
        # Extract the chunks and corresponding page numbers
        chunks, page_numbers = zip(*chunks_with_page_numbers)
        
        # Create Document objects with the chunk content and metadata (page number)
        documents = [
            Document(page_content=chunk, metadata={"page_number": page_number})
            for chunk, page_number in zip(chunks, page_numbers)
        ]
        
        # Create a FAISS index from the documents
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        
        # Save the vector store locally for later use
        vector_store.save_local("faiss_index")
        
        return chunks_with_page_numbers  # Return the chunks with page numbers to use later for display
    
    except Exception as e:
        st.error(f"Error while creating vector store: {e}")  # Display error if something goes wrong
        return []

# Function to create a conversational chain with a generative model for question answering
def get_conversational_chain():
    # Define the prompt template for the question answering model
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize the Google Generative AI model for conversation
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Create a prompt template for the question-answering chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the question-answering chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain  # Return the chain for use in answering questions

# Function to handle user input and provide answers
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use embedding model for queries
    
    # Load the FAISS index with deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search for the user question in the vector store
    docs = new_db.similarity_search(user_question)

    # Initialize variables to track the relevant page number and highest score
    relevant_page_number = None
    highest_score = -1  # Start with a low score to find the highest

    # Loop through the retrieved documents to find the one with the highest score
    for doc in docs:
        score = doc.metadata.get('score', None)  # Get the score (if available) from the document's metadata
        
        # If no score is found, take the first document's page number
        if score is None:
            relevant_page_number = doc.metadata['page_number']
            break  # Stop once a document with no score is found

        # If the score is higher than the current highest score, update it
        if score > highest_score:
            highest_score = score
            relevant_page_number = doc.metadata['page_number']

    # Get the conversational chain for question answering
    chain = get_conversational_chain()

    # Use the chain to generate a response for the question
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    # Display the generated response and the relevant page number
    st.write(f"Reply: {response['output_text']}")
    st.write(f"Page Number: {relevant_page_number}")

# Main function to control the Streamlit interface
def main():
    st.set_page_config("Chat PDF")  # Set the page configuration for Streamlit
    st.header("Chat with PDF")  # Display header in the app

    user_question = st.text_input("Ask a Question from the PDF Files")  # Text input for user question

    # If a question is entered, process it
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")  # Sidebar title
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Button to submit and process the uploaded PDFs
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):  # Show a loading spinner while processing
                    try:
                        # Extract text from the uploaded PDFs with page numbers
                        text_with_page_numbers = get_pdf_text(pdf_docs)
                        
                        # Split the extracted text into smaller chunks
                        chunks_with_page_numbers = get_text_chunks(text_with_page_numbers)
                        
                        # Create and save the FAISS vector store for later use
                        get_vector_store(chunks_with_page_numbers)
                        st.success("Processing completed successfully!")  # Show success message
                    except Exception as e:
                        st.error(f"Error during processing: {e}")  # Display error message if processing fails
            else:
                st.warning("Please upload PDF files before processing.")  # Warning if no PDFs are uploaded

# Run the main function to start the app
if __name__ == "__main__":
    main()
