from py.daily_fetch import * 
from py.handle_files import *
from py.ingest import *
from py.chroma_db_storage import *
from py.sentiment_analysis import *

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any

# # Collect Data    
# social_media_data = collect_social_media_and_news_data(get_company_list(), get_start_date(), get_end_date())

# # Save collected data to Files
# create_files(social_media_data)

# # Ingest - prepare the data for LLM feeding
# ingested_data = ingest_data()

# # Save Ingested Data
# save_ingested_data(ingested_data)

# # Sentiment analyse Ingested Data
# # analysed_data = sentiment_analyse(get_ingested_data())
# analysed_data = sentiment_analyse(ingested_data)

# # Save Analysed Data
# save_analysed_data(analysed_data)

# Retrieve data from Pickle file
pickled_data = get_analysed_data()
print(len(pickled_data))
        
# Delete existing ChromaDB 
delete_existingchromaDBs()
# Initialize the ChromaDBStorage
chroma_storage = ChromaDBStorage("sentiment_analysed_data")
chroma_storage.save_documents(pickled_data)

