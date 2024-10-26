import warnings
from py.data_fetch import * 
from py.handle_files import *
from py.db_storage import *

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
warnings.filterwarnings("ignore")

stock="nse"
# stock="nasdaq"

# Collect Data    
data_fetch = DataFetch()
data_fetch.load_company_list("Stock Sentiment Analysis/Resources/"+stock+"_companies.csv")
social_media_data = data_fetch.collect_data()

# Save collected data to Files
create_files(social_media_data)

# Fetch saved Social Media Data
social_media_document = fetch_social_media_data()
print(len(social_media_document))

# Samples `n` entries for each unique `"platform"` and `"company"` metadata combination from the input `Document[]`.
social_media_document_samples = sample_documents(social_media_document, 20)
print(len(social_media_document_samples))

# Delete and clear any ChromaDB databases
clear_db()

# Initialise ChromaDB Database
chroma_db = DBStorage()

# Create chunks and embeddings in the database
FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_HD")
chroma_db.embed_vectors(social_media_document_samples, FAISS_DB_PATH)


        


