import os
import pickle
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import warnings
import shutil


warnings.filterwarnings("ignore")

CHROMA_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "chroma_db")


def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks 


def create_embeddings_chroma(chunks, persist_directory=CHROMA_PATH):
    from langchain_community.vectorstores import Chroma
 
    # Instantiate an embedding model from Azure OpenAI
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key="3a93fe43c4534148a7a193412e29a321",
        api_version="2024-02-01",
        azure_endpoint="https://ey-openai.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
    )

    print("here A")
    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store

def chromaDBStorage(data):
    # Function to check if a directory exists
    # print(CHROMA_PATH)
    if os.path.isdir(CHROMA_PATH):
        # os.rmdir(CHROMA_PATH)
        # shutil.rmtree(CHROMA_PATH)
        shutil.rmtree(CHROMA_PATH, onexc=lambda func, path, exc: os.chmod(path, 0o777))
        print('deleted ' + CHROMA_PATH)
        
    # Splitting the document into chunks
    chunks = chunk_data(data)

    # print(chunks)
    # Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
    vector_store = create_embeddings_chroma(chunks)
    
    print("Here B")
    
    # Save the list to a file
    with open('Stock Sentiment Analysis/files/vector_store.pkl', 'wb') as file:
        pickle.dump(vector_store, file)
    
    print(vector_store)

