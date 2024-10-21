import os
import pickle
from typing import List, Dict, Any
from datetime import datetime
from uuid import uuid4
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import warnings
import shutil

warnings.filterwarnings("ignore")

CHROMA_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "chroma_db")

class SentimentAnalysisChromaStorage:
    def __init__(self, collection_name: str = "stock_sentiment_analysis"):
        self.collection_name = collection_name
        self.persist_directory = CHROMA_PATH
        self.embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )
        self.vector_store = None

    def chunk_data(self, data: List[Document], chunk_size: int = 256) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        return text_splitter.split_documents(data)

    def create_embeddings_chroma(self, chunks: List[Document]) -> Chroma:
        return Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

    def store_data(self, data: List[Document]) -> None:
        if os.path.isdir(self.persist_directory):
            shutil.rmtree(self.persist_directory, onexc=lambda func, path, exc: os.chmod(path, 0o777))
            print(f'Deleted existing {self.persist_directory}')

        chunks = self.chunk_data(data)
        self.vector_store = self.create_embeddings_chroma(chunks)
        self.vector_store.persist()
        print(f"Stored {len(chunks)} chunks in Chroma DB.")

        # Save the vector store to a file
        with open('Stock Sentiment Analysis/files/vector_store.pkl', 'wb') as file:
            pickle.dump(self.vector_store, file)
        print("Vector store saved to file.")

    def query_data(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please store data first.")
        
        results = self.vector_store.similarity_search_with_score(query_text, k=n_results)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            } for doc, score in results
        ]

    def get_data_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please store data first.")
        
        results = self.vector_store.get(
            where={
                "timestamp": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }
        )
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            } for doc in results
        ]

    def update_data(self, id: str, new_content: str, new_metadata: Dict[str, Any]) -> None:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please store data first.")
        
        self.vector_store.update(
            ids=[id],
            documents=[new_content],
            metadatas=[new_metadata]
        )
        print(f"Updated document with ID: {id}")

    def delete_data(self, id: str) -> None:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please store data first.")
        
        self.vector_store.delete(ids=[id])
        print(f"Deleted document with ID: {id}")

# Example usage
if __name__ == "__main__":
    # Initialize the SentimentAnalysisChromaStorage
    chroma_storage = SentimentAnalysisChromaStorage()
    
    # Example documents
    sample_docs = [
        Document(page_content="NASDAQ stock prices are rising.", metadata={"source": "twitter", "sentiment": "positive", "timestamp": datetime.now().isoformat()}),
        Document(page_content="New tech IPO announced on NASDAQ.", metadata={"source": "facebook", "sentiment": "neutral", "timestamp": datetime.now().isoformat()})
    ]
    
    # Store the documents
    chroma_storage.store_data(sample_docs)
    
    # Query the data
    query_results = chroma_storage.query_data("NASDAQ stock")
    print("Query Results:")
    for result in query_results:
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Score: {result['score']}")
        print("---")
    
    # Get data by date range
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.now()
    date_range_results = chroma_storage.get_data_by_date_range(start_date, end_date)
    print("Date Range Results:")
    for result in date_range_results:
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print("---")