from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import pandas as pd
import json
import re
from datetime import datetime


def clean_text(text: str) -> str:
    """
    Remove HTML tags, special characters, and extra whitespace from text.
    
    Args:
    text (str): The input text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z.A-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase.
    
    Args:
    text (str): The input text to be normalized.
    
    Returns:
    str: The normalized text.
    """
    return text.lower()


def clean_and_preprocess(data):
    cleaned_data = []
    for item in data:
        # Remove HTML tags, special characters, etc.
        cleaned_text = clean_text(str(item["page_content"]))
        # Normalize text (lowercase, remove extra whitespace)
        normalized_text = normalize_text(cleaned_text)
        # Create a new Document with cleaned and normalized text
        # cleaned_data.append(Document(page_content=normalized_text))
        cleaned_data.append(Document(page_content=normalized_text, 
                                     metadata={"platform":item["platform"],"company":item["company"]}))
    return cleaned_data

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def add_metadata(documents: List[Document]) -> List[Document]:
    """
    Add additional metadata to the documents.
    
    Args:
    documents (List[Document]): List of Document objects.
    
    Returns:
    List[Document]: List of Document objects with additional metadata.
    """
    for doc in documents:
        doc.metadata['ingestion_timestamp'] = datetime.now().isoformat()
        doc.metadata['word_count'] = len(doc.page_content.split())
    return documents

def validate_data(documents: List[Document]) -> List[Document]:
    """
    Validate the data and remove any invalid documents.
    
    Args:
    documents (List[Document]): List of Document objects to be validated.
    
    Returns:
    List[Document]: List of valid Document objects.
    """
    valid_documents = []
    for doc in documents:
        if len(doc.page_content.split()) > 5:  # Ensure document has more than 5 words
            if all(key in doc.metadata for key in ['platform', 'company']):  # Ensure required metadata is present
                valid_documents.append(doc)
    return valid_documents


def ingest_data():
    # social_media_data = pd.read_pickle("social_media_data.pkl")
    with open('Stock Sentiment Analysis/files/social_media_data.json', 'r') as file:
        data = json.load(file)
        
    cleaned_data = clean_and_preprocess(data)
    split_data = split_documents(cleaned_data)
    enriched_data = add_metadata(split_data)
    valid_data = validate_data(enriched_data)
    return valid_data


# ingested_data = ingest_data(social_media_data)
# print(ingested_data)
