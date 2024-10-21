# tumblr auth:
from datetime import datetime, timedelta
import math
import os
import pandas as pd
import tweepy

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


print(os.getenv("OPR_TST"))
print(os.getenv("TWITTER_API_KEY"))
print(os.getenv("TWITTER_API_SECRET"))
print(os.getenv("TWITTER_ACCESS_TOKEN"))
print(os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
      
twitter_auth = tweepy.OAuthHandler(os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET"))
twitter_auth.set_access_token(os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
twitter_api = tweepy.API(twitter_auth)



def collect_twitter_data(api, query, start_date, end_date):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en",
                               since=start_date, until=end_date).items(5):
        tweets.append(tweet._json)
    return tweets

def collect_social_media_data(company_list, start_date, end_date):
    # Set up API clients
    all_data = []
    for company in company_list:

        twitter_data = collect_twitter_data(twitter_api, company, start_date, end_date)
        

        all_data.extend([
            {"platform": "Twitter", "company": company, "data": item} for item in twitter_data
        ])
    return all_data


social_media_data = collect_social_media_data(company_list, start_date_t, end_date_t)

print(social_media_data)