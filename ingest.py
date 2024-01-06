#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Tuple
from langchain.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
from dotenv import load_dotenv

load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50

def split_documents_into_chunks(documents: List[Document], chunk_size: int):
    """
    Split a list of langchain.Document objects into chunks of text with a maximum size of 'chunk_size' tokens each.
    """
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for doc in documents:
        text = doc.page_content
        tokens = text.split()  # Split the text into individual tokens
        for token in tokens:
            current_chunk += token + " "
            current_chunk_size += 1

            if current_chunk_size >= chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_size = 0

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def train_model(documents: List[Document]):
    """
    Train the language model on the given list of langchain.Document objects.
    """
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=HuggingFaceEmbeddings(embeddings_model_name).embed,
    )
    # Split the list of langchain.Document objects into chunks of tokens
    chunks = split_documents_into_chunks(documents, chunk_size=chunk_size)

    # Train the language model on the chunks
    for chunk in chunks:
        db.add_documents([chunk], metadata={'source': 'training_data'})

def process_documents(content_url_pairs: List[Tuple[str, str]]) -> List[Document]:
    """
    Process the given content and URL pairs and create langchain.Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []

    for content, url in content_url_pairs:
        try:
            # Ensure that the page_content is a string
            page_content = str(content)

            # Create a langchain.Document object with the correct field name
            doc = Document(page_content=page_content, metadata={'source': url})

            # Split the langchain.Document into chunks using the text splitter
            chunks = text_splitter.split_documents([doc])

            # Add the chunks to the list of documents
            documents.extend(chunks)
        except Exception as e:
            print(f"Error occurred while splitting document: {e}")

    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    return documents

def ingest_data(content: str, url: str):
    # Create a list of content and URL pairs
    content_url_pairs = [(content, url)]

    # Process the content and URL pairs to create langchain.Document objects
    documents = process_documents(content_url_pairs)

    # Check if any document has content less than 10 characters
    if any(len(doc.page_content) < 10 for doc in documents):
        print("Some documents have content less than 10 characters. Exiting.")
        return

    # Train the model with the documents
    train_model(documents)

# Use the content from the text file for ingestion
def ingest_from_text_file(file_path: str, url: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    ingest_data(content, url)

# Example usage:
# Replace 'path/to/your/textfile.txt' with the actual file path containing the content.
ingest_from_text_file('training_data/textfile.txt', 'https://www.google.com')
