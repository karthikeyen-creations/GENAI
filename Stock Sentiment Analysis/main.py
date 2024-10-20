from daily_fetch import * 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
    
social_media_data = collect_social_media_and_news_data(get_company_list(), get_start_date(), get_end_date())

print(social_media_data)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(social_media_data)
df.head()

# Exporting the data to a CSV file
file_path = "Stock Sentiment Analysis/social_media_data.csv"
df.to_csv(file_path, index=False)
    
df.to_pickle("Stock Sentiment Analysis/social_media_data.pkl")