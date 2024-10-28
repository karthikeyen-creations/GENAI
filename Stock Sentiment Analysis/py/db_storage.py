import os
import warnings
import shutil
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from typing import List, Optional, Dict, Any
from langchain.schema import Document
import chromadb
# from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain_community.vectorstores import FAISS



warnings.filterwarnings("ignore")
CHROMA_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "chroma_db")
CHROMA_DB_PATHH = os.path.join(os.getcwd(), "chroma_db")
# FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_index")
tesla_10k_collection = 'tesla-10k-2019-to-2023'
embedding_model = ""
# embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

def clear_db(hugg = False):
    check_and_delete(CHROMA_DB_PATH)
    if hugg:
        check_and_delete(CHROMA_DB_PATHH)
    # check_and_delete(FAISS_DB_PATH)

class DBStorage:
    def __init__(self, hugg = False):
        self.hugg = hugg    
        self.CHROMA_PATH = CHROMA_DB_PATH
        if self.hugg:
            self.CHROMA_PATH = CHROMA_DB_PATHH
        self.vector_store = None
        self.client = chromadb.PersistentClient(path=self.CHROMA_PATH)
        print(self.client.list_collections())
        self.collection = self.client.get_or_create_collection(name=tesla_10k_collection)
        print(self.collection.count())

    def chunk_data(self, data, chunk_size=10000):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        return text_splitter.split_documents(data)

    def create_embeddings(self, chunks):
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )
        
        self.vector_store = Chroma.from_documents(documents=chunks, 
                                                #   embedding=embeddings, 
                                                  embedding=embedding_model, 
                                                  collection_name=tesla_10k_collection, 
                                                  persist_directory=self.CHROMA_PATH)
        print("Here B")
        self.collection = self.client.get_or_create_collection(name=tesla_10k_collection)
        print("here"+str(self.collection.count()))
        # return self.vector_store
    
    def create_vector_store(self, chunks):
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )
        return FAISS.from_documents(chunks, embedding=embeddings)
        # vector_store.save_local(FAISS_DB_PATH)
        
    
    def load_embeddings(self):
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )

        self.vector_store = Chroma(collection_name=tesla_10k_collection, 
                                   persist_directory=self.CHROMA_PATH, 
                                #    embedding_function=embeddings
                                   embedding_function=embedding_model
                                   )
        print("loaded vector store: ")
        print(self.vector_store)
        # return self.vector_store

    def load_vectors(self,FAISS_DB_PATH):
        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )

        self.vector_store = FAISS.load_local(folder_path=FAISS_DB_PATH, 
                                            embeddings=embeddings, 
                                            allow_dangerous_deserialization=True)


    
    def fetch_documents(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        results = self.collection.get(
            where=metadata_filter,
            include=["documents", "metadatas"],
        )

        documents = []
        for content, metadata in zip(results['documents'][0], results['metadatas'][0]):
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    
    def get_context_for_query(self, question, k=3):
        print(self.vector_store)
        # if not self.vector_store:
        #     raise ValueError("Vector store not initialized. Call create_embeddings() or load_embeddings() first.")

        # relevant_document_chunks=self.fetch_documents({"company": question})

        # retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        # relevant_document_chunks = retriever.get_relevant_documents(question)
        
        relevant_document_chunks = self.vector_store.similarity_search(question)
        # chain = get_conversational_chain(models.llm)
        # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        # print(response)
        
        print(relevant_document_chunks)
        context_list = [d.page_content for d in relevant_document_chunks]
        context_for_query = ". ".join(context_list)
        print("context_for_query: "+ str(len(context_for_query)))

        return context_for_query
    
    # def ask_question(self, question, k=3):
    #     if not self.vector_store:
    #         raise ValueError("Vector store not initialized. Call create_embeddings() or load_embeddings() first.")

    #     llm = AzureChatOpenAI(
    #         temperature=0,
    #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #         model=os.getenv("AZURE_OPENAI_MODEL_NAME")
    #     )

    #     retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    #     chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
    #     return chain.invoke(question)
    
    def embed_vectors(self,social_media_document,FAISS_DB_PATH):
        print("here A")
        chunks = self.chunk_data(social_media_document)
        print(len(chunks))
        # self.create_embeddings(chunks)
        vector_store = self.create_vector_store(chunks)
        check_and_delete(FAISS_DB_PATH, self.hugg)
        vector_store.save_local(FAISS_DB_PATH)

def check_and_delete(PATH, hugg=False):
    if os.path.isdir(PATH):
        if hugg:
            shutil.rmtree(PATH)
        else:
            shutil.rmtree(PATH, onexc=lambda func, path, exc: os.chmod(path, 0o777))
            
        print(f'Deleted {PATH}')
    


# Usage example
if __name__ == "__main__":
    qa_system = DBStorage()

    # Load and process document
    social_media_document = []
    chunks = qa_system.chunk_data(social_media_document)

    # Create embeddings
    qa_system.create_embeddings(chunks)

    # # Ask a question
    # question = 'Summarize the whole input in 150 words'
    # answer = qa_system.ask_question(question)
    # print(answer)