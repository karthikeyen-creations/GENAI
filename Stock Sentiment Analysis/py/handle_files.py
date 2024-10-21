

import json
import os
import pickle
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