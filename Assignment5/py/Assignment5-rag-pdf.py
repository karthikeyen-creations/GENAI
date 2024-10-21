import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import warnings
import shutil

warnings.filterwarnings("ignore")

CHROMA_PATH = os.path.join(os.getcwd(), "Assignment5", "py", "chroma_db")

# If getting upgrade warning, run the following from command line:
# chroma utils vacuum --path chroma_db
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data
  

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

    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store


def load_embeddings_chroma(persist_directory=CHROMA_PATH):
    from langchain.vectorstores import Chroma

    # Instantiate the same embedding model used during creation
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key="3a93fe43c4534148a7a193412e29a321",
        api_version="2024-02-01",
        azure_endpoint="https://ey-openai.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
    )

    # Load the Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 

    return vector_store  # Return the loaded vector store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA

    llm = AzureChatOpenAI(temperature=0,
                      api_key="3a93fe43c4534148a7a193412e29a321",
                      api_version="2024-02-01",
                      azure_endpoint="https://ey-openai.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview",
                      model="gpt-35-turbo"
            )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer

# Function to check if a directory exists
if os.path.isdir(CHROMA_PATH):
    # os.rmdir(CHROMA_PATH)
    shutil.rmtree(CHROMA_PATH, onexc=lambda func, path, exc: os.chmod(path, 0o777))
    # shutil.rmtree(CHROMA_PATH)
    print('deleted ' + CHROMA_PATH)
    
# Loading the pdf document into LangChain 
data = load_document(os.path.join( os.getcwd(), 'Assignment5','py','files', 'Personal Note.pdf'))

# print(data)
# Load the document from Wikipedia
# data = load_from_wikipedia("Amitabh Bacchan")

# Splitting the document into chunks
chunks = chunk_data(data)

print(chunks)
# Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
vector_store = create_embeddings_chroma(chunks)

# Asking questions
q = 'Summarize the whole input in 150 words'
answer = ask_and_get_answer(vector_store, q)
print(answer)