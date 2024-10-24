import warnings
from py.daily_fetch import * 
from py.handle_files import *
from py.chroma_db_storage import *

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

# # Collect Data    
# social_media_data = collect_social_media_and_news_data(get_company_list(), get_start_date(), get_end_date())

# # Save collected data to Files
# create_files(social_media_data)

# Fetch saved Social Media Data
social_media_document = fetch_social_media_data()
print(len(social_media_document))

# Samples `n` entries for each unique `"platform"` and `"company"` metadata combination from the input `Document[]`.
social_media_document_samples = sample_documents(social_media_document, 10)
print(len(social_media_document_samples))

# Delete and clear any ChromaDB databases
clear_chroma_db()

# Initialise ChromaDB Database
chroma_db = ChromaDBStorage()

# Create chunks and embeddings in the database
chroma_db.embed_vectors(social_media_document_samples)


# # Ingest - prepare the data for LLM feeding
# ingested_data = ingest_data()

# # Save Ingested Data
# save_ingested_data(ingested_data)

# # Sentiment analyse Ingested Data
# # analysed_data = sentiment_analyse(get_ingested_data())
# analysed_data = sentiment_analyse(ingested_data)

# # Save Analysed Data
# save_analysed_data(analysed_data)

# # Retrieve data from Pickle file
# pickled_data = get_analysed_data()
# print(len(pickled_data))
        


