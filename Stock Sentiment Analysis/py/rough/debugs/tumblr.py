# tumblr auth:
from datetime import datetime, timedelta
import math
import os
import pandas as pd
import pytumblr

def load_company_list(file_path):
    return pd.read_csv(file_path)['company_ticker'].tolist()

company_list = load_company_list('Stock Sentiment Analysis/nasdaq_companies.csv')
print(company_list)
end_date = datetime.now() - timedelta(hours=5, minutes=30)
start_date = end_date - timedelta(days=1) 

end_date_t=math.floor(end_date.timestamp())
start_date_t=math.floor(start_date.timestamp())

print(start_date_t)
print(end_date_t)

tumblr_client = pytumblr.TumblrRestClient(
    os.getenv("TUMBLR_CONSUMER_KEY"),
    os.getenv("TUMBLR_CONSUMER_SECRET"),
    os.getenv("TUMBLR_OAUTH_TOKEN"),
    os.getenv("TUMBLR_OAUTH_SECRET")
)
# print(tumblr_client.info())
# qry='AAPL'
# print(tumblr_client.tagged(qry))

def collect_tumblr_data(client, query, start_date, end_date):
    print(query)
    posts = client.tagged(query)
    # print(posts) 
    # filtered_posts = [post for post in posts if start_date <= post['timestamp'] <= end_date]
    # for post in posts:
    #     print(post['timestamp'])
    return posts



def collect_social_media_data(company_list, start_date, end_date):
    # Set up API clients
    all_data = []
    for company in company_list:

        tumblr_data = collect_tumblr_data(tumblr_client, company, start_date, end_date)
        

        all_data.extend([
            {"platform": "Tumblr", "company": company, "data": item} for item in tumblr_data
        ])
    
    return all_data

social_media_data = collect_social_media_data(company_list, start_date_t, end_date_t)

print(social_media_data)