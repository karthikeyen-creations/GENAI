

from datetime import datetime
import json
import os
import pickle
from typing import List
from langchain.schema import Document
import pandas as pd

def create_files(social_media_data):
    folder_path = 'Stock Sentiment Analysis/files'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Save dictionary to a file
    with open(folder_path+'/social_media_data.json', 'w') as f:
        json.dump(social_media_data, f)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(social_media_data)
    df.head()

    # Exporting the data to a CSV file
    file_path = folder_path+"/social_media_data.csv"
    df.to_csv(file_path, index=False)
        
    df.to_pickle(folder_path+"/social_media_data.pkl")

def fetch_social_media_data():
        with open('Stock Sentiment Analysis/files/social_media_data.json', 'r') as file:
            data = json.load(file)
        social_media_document = []
        for item in data:
            social_media_document.append(Document(
                page_content=str(item["page_content"]), 
                metadata={"platform":item["platform"],
                          "company":item["company"],
                          "ingestion_timestamp":datetime.now().isoformat(),
                          "word_count":len(item["page_content"]["content"])
                          }))
        return social_media_document
        
def save_ingested_data(ingested_data):
    # Save the list to a file
    with open('Stock Sentiment Analysis/files/ingested_data.pkl', 'wb') as file:
        pickle.dump(ingested_data, file)

def save_analysed_data(analysed_data):
    # Save the list to a file
    with open('Stock Sentiment Analysis/files/analysed_data.pkl', 'wb') as file:
        pickle.dump(analysed_data, file)

def get_ingested_data():
    # Load the list from the file
    with open('Stock Sentiment Analysis/files/ingested_data.pkl', 'rb') as file:
        loaded_documents = pickle.load(file)
    return loaded_documents

def get_analysed_data():
    # Load the list from the file
    with open('Stock Sentiment Analysis/files/analysed_data.pkl', 'rb') as file:
        loaded_documents = pickle.load(file)
    return loaded_documents

def sample_documents(documents: List[Document], n: int) -> List[Document]:
    """
    Samples `n` entries for each unique `"platform"` and `"company"` metadata combination from the input `Document[]`.

    Args:
        documents (List[Document]): The input list of `Document` objects.
        n (int): The number of entries to sample for each unique metadata combination.

    Returns:
        List[Document]: A new list of `Document` objects, with `n` entries per unique metadata combination.
    """
    # Create a dictionary to store the sampled documents per metadata combination
    sampled_docs = {}

    for doc in documents:
        combo = (doc.metadata["platform"], doc.metadata["company"])
        if combo not in sampled_docs:
            sampled_docs[combo] = []
        
        # Add the document to the list for its metadata combination, up to n entries
        if len(sampled_docs[combo]) < n:
            sampled_docs[combo].append(doc)

    # Flatten the dictionary into a single list
    return [doc for docs in sampled_docs.values() for doc in docs]

def to_documents(data) -> List[Document]:
    social_media_document = []
    for item in data:
        social_media_document.append(Document(
            page_content=str(item["page_content"]), 
            metadata={"platform":item["platform"],
                        "company":item["company"],
                        "ingestion_timestamp":datetime.now().isoformat(),
                        "word_count":len(item["page_content"]["content"])
                        }))
    return social_media_document