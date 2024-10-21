import shutil
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Optional, Dict, Any
from langchain.schema import Document
import numpy as np
import certifi
import httpx 
import ssl

persist_directory="./Stock Sentiment Analysis/chroma_db"

# # Create an SSL context using certifi
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# # Monkey-patch httpx to use the new SSL context
# httpx._default_ssl_context = ssl_context

# # Globally disable SSL verification
# httpx._default_ssl_context.check_hostname = False
# httpx._default_ssl_context.verify_mode = ssl.CERT_NONE

# # Create an SSL context using certifi
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# # Create a custom HTTP transport with the custom SSL context
# transport = httpx.HTTPTransport(ssl_context=ssl_context)
# # Create a custom client that uses the transport
# http_client = httpx.Client(transport=transport)

class ChromaDBStorage:
    def __init__(self, collection_name: str, persist_directory: str = persist_directory):
        # # Disable SSL verification (use with caution, not recommended for production)
        # http_client = httpx.Client(verify=False)
        # # Use a custom HTTP client with SSL certificates verification using certifi
        # http_client = httpx.Client(verify=certifi.where())
        # self.client = chromadb.PersistentClient(path=persist_directory, http_client=http_client)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_function = None


    def set_embedding_function(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
 
    def save_documents(self, documents: List[Document]):
        ids = [str(i) for i in range(len(documents))]  # Generate IDs if not present
        contents = [doc.page_content for doc in documents]
        # metadatas = [doc.metadata for doc in documents if doc.metadata]
        metadatas = metadatas = [self._sanitize_metadata(doc.metadata) for doc in documents if doc.metadata]
        platforms = [doc.metadata["platform"] for doc in documents if doc.metadata]
        companys = [doc.metadata["company"] for doc in documents if doc.metadata]
        ingestion_timestamps = [doc.metadata["ingestion_timestamp"] for doc in documents if doc.metadata]
        word_count = [doc.metadata["word_count"] for doc in documents if doc.metadata]
        positives = [doc.metadata["positive"] for doc in documents if doc.metadata]
        neutrals = [doc.metadata["neutral"] for doc in documents if doc.metadata]
        negatives = [doc.metadata["negative"] for doc in documents if doc.metadata]
         

        embeddings = None
        if self.embedding_function:
            embeddings = self.embedding_function(contents)

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas if metadatas else None,
            embeddings=embeddings
        )
        self.client.persist()
        print(f"Added {len(documents)} documents to ChromaDB collection '{self.collection.name}'")

    def fetch_documents(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        results = self.collection.query(
            query_texts=None,
            where=metadata_filter,
            include=["documents", "metadatas"]
        )

        documents = []
        for content, metadata in zip(results['documents'][0], results['metadatas'][0]):
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        def sanitize_value(value):
            if isinstance(value, np.float32):
                return float(value)
            elif isinstance(value, (str, int, float, bool)):
                return value
            else:
                return str(value)

        return {k: sanitize_value(v) for k, v in metadata.items()}
    
    def fetch_documents_by_multiple_metadata(self, metadata_filters: List[Dict[str, Any]]) -> List[Document]:
        all_documents = []
        for filter in metadata_filters:
            documents = self.fetch_documents(filter)
            all_documents.extend(documents)
        return all_documents


def delete_existingchromaDBs():
        if os.path.isdir(persist_directory):
            shutil.rmtree(persist_directory, onexc=lambda func, path, exc: os.chmod(path, 0o777))
            print(f'Deleted existing {persist_directory}')

    
# Example usage:
if __name__ == "__main__":
    import os

    # Specify the directory for ChromaDB
    chroma_directory = os.path.join(os.getcwd(), "my_chroma_db")

    # Create some sample documents
    docs = [
        Document(page_content="This is a document about Python", metadata={"topic": "programming", "language": "Python"}),
        Document(page_content="This is a document about JavaScript", metadata={"topic": "programming", "language": "JavaScript"}),
        Document(page_content="This is a document about machine learning", metadata={"topic": "AI", "subtopic": "machine learning"}),
        Document(page_content="This is a document about deep learning", metadata={"topic": "AI", "subtopic": "deep learning"})
    ]

    # Initialize the ChromaDBStorage with a specific storage location
    saver = ChromaDBStorage("my_collection", persist_directory=chroma_directory)

    # Save the documents
    saver.save_documents(docs)

    # Fetch documents about programming
    programming_docs = saver.fetch_documents({"topic": "programming"})
    print(f"Found {len(programming_docs)} documents about programming")

    # Fetch documents about Python or AI
    multi_filter_docs = saver.fetch_documents_by_multiple_metadata([
        {"language": "Python"},
        {"topic": "AI"}
    ])
    print(f"Found {len(multi_filter_docs)} documents about Python or AI")