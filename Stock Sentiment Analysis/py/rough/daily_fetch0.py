import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import tweepy
import praw
import googleapiclient.discovery
import pytumblr
import math
from gnews import GNews
import requests
from bs4 import BeautifulSoup
import time

def load_company_list(file_path):
    return pd.read_csv(file_path)['company_ticker'].tolist()

company_list = load_company_list('Stock Sentiment Analysis/nasdaq_companies.csv')
end_date = datetime.now()
start_date = end_date - timedelta(days=1)

def get_company_list():
    return company_list

def get_end_date():
    return end_date

def get_start_date():
    return start_date

#Social Media
def collect_tumblr_data(client, query, start_date, end_date):
    end_date_t=math.floor(end_date.timestamp())
    start_date_t=math.floor(start_date.timestamp())
    posts = client.tagged(query)
    # filtered_posts = [post for post in posts if start_date_t <= post['timestamp'] <= end_date_t]
    # return filtered_posts
    return posts


def collect_twitter_data(api, query, start_date, end_date):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en",
                               since=start_date, until=end_date).items(100):
        tweets.append(tweet._json)
    return tweets

def collect_reddit_data(reddit, query, start_date, end_date):
    subreddit = reddit.subreddit("all")
    posts = []
    for post in subreddit.search(query, sort="new", time_filter="day"):
        if start_date <= datetime.fromtimestamp(post.created_utc) <= end_date:
            posts.append({
                "title": post.title,
                "content": post.selftext,
                "author": post.author.name if post.author else "[deleted]",
                "created_utc": post.created_utc,
                "score": post.score,
                "num_comments": post.num_comments,
                "url": post.url
            })
    return posts

def collect_youtube_data(youtube, query, start_date, end_date):
    request = youtube.search().list(
        q=query,
        type="video",
        part="id,snippet",
        maxResults=50,
        publishedAfter=start_date.isoformat() + "Z",
        publishedBefore=end_date.isoformat() + "Z"
    )
    response = request.execute()
    return response['items']

# News sites

def collect_google_news(query, start_date, end_date):
    google_news = GNews(language='en', country='US', start_date=start_date, end_date=end_date)
    news_articles = google_news.get_news(query)
    return news_articles

def collect_financial_times(query, start_date, end_date):
    # Note: This is a simplified example. You might need to handle pagination, authentication, etc.
    url = f"https://www.ft.com/search?q={query}&dateTo={end_date.strftime('%Y-%m-%d')}&dateFrom={start_date.strftime('%Y-%m-%d')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('div', class_='o-teaser__content')
    return [{'title': a.find('div', class_='o-teaser__heading').text.strip(),
             'description': a.find('p', class_='o-teaser__standfirst').text.strip() if a.find('p', class_='o-teaser__standfirst') else '',
             'link': 'https://www.ft.com' + a.find('a')['href']} for a in articles]

def collect_bloomberg(query, start_date, end_date):
    # Similar to Financial Times, but for Bloomberg
    url = f"https://www.bloomberg.com/search?query={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('div', class_='storyItem__aaf871c1')
    return [{'title': a.find('a', class_='headline__3a97424d').text.strip(),
             'description': a.find('p', class_='summary__483358e1').text.strip() if a.find('p', class_='summary__483358e1') else '',
             'link': a.find('a', class_='headline__3a97424d')['href']} for a in articles]

def collect_reuters(query, start_date, end_date):
    base_url = "https://www.reuters.com/site-search/"
    articles = []
    page = 1
    
    while True:
        url = f"{base_url}?blob={query}&page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # print(soup)
        
        results = soup.find_all('li', class_='search-result__item')
        
        if not results:
            break
        
        flg002=0
        
        for result in results:
            date_elem = result.find('time', class_='search-result__timestamp')
            if date_elem:
                date = datetime.strptime(date_elem['datetime'], "%Y-%m-%dT%H:%M:%SZ")
                print(date)
                if start_date <= date <= end_date:
                    title = result.find('h3', class_='search-result__headline').text.strip()
                    link = "https://www.reuters.com" + result.find('a')['href']
                    description = result.find('p', class_='search-result__excerpt').text.strip()
                    
                    articles.append({
                        'title': title,
                        'link': link,
                        'description': description,
                        'date': date.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    flg002=1
                elif date < start_date:
                    return articles  # Stop if we've gone past our date range

        
        page += 1
        time.sleep(1)  # Be respectful with request frequency
        if flg002==0:
            return articles
    
    return articles


def collect_wsj(query, start_date, end_date):
    base_url = "https://www.wsj.com/search"
    articles = []
    page = 1
    
    while True:
        params = {
            'query': query,
            'isToggleOn': 'true',
            'operator': 'AND',
            'sort': 'date-desc',
            'duration': 'custom',
            'startDate': start_date.strftime('%Y/%m/%d'),
            'endDate': end_date.strftime('%Y/%m/%d'),
            'source': 'wsjie,wsjblogs,wsjvideo,interactivemedia,wsjsitesrch,wsjpro',
            'page': page
        }
        
        response = requests.get(base_url, params=params)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = soup.find_all('article', class_='WSJTheme--story--XB4V2mLz')
        
        # print(results)
        
        if not results:
            break
        
        flg01=0
        
        for result in results:
            date_elem = result.find('p', class_='WSJTheme--timestamp--22sfkNDv')
            if date_elem:
                date = datetime.strptime(date_elem.text.strip(), "%B %d, %Y")
                if start_date <= date <= end_date:
                    title_elem = result.find('h3', class_='WSJTheme--headline--7VCzo7Ay')
                    title = title_elem.text.strip() if title_elem else "No title"
                    link = "https://www.wsj.com" + result.find('a')['href']
                    description_elem = result.find('p', class_='WSJTheme--summary--lmOXEsbN')
                    description = description_elem.text.strip() if description_elem else "No description"
                    
                    articles.append({
                        'title': title,
                        'link': link,
                        'description': description,
                        'date': date.strftime("%Y-%m-%d")
                    })
                    flg01 = 1
                elif date < start_date:
                    return articles  # Stop if we've gone past our date range
        
        page += 1
        time.sleep(1)  # Be respectful with request frequency
        if flg01==0:
            return articles
    
    return articles


def search_news(query: str, days: int = 14) -> List[Dict]:
    """Search news using Serper API"""
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": f"{query} company news",
        "num": 500,
        "dateRestrict": f"d{days}"
    }
    
    response = requests.post(
        "https://google.serper.dev/news",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json().get("news", [])
    return []

def collect_social_media_and_news_data(company_list, start_date, end_date):
    # Set up API clients
    tumblr_client = pytumblr.TumblrRestClient(
    os.getenv("TUMBLR_CONSUMER_KEY"),
    os.getenv("TUMBLR_CONSUMER_SECRET"),
    os.getenv("TUMBLR_OAUTH_TOKEN"),
    os.getenv("TUMBLR_OAUTH_SECRET")
    )
    
    twitter_auth = tweepy.OAuthHandler(os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET"))
    twitter_auth.set_access_token(os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
    twitter_api = tweepy.API(twitter_auth)

    reddit = praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                         client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                         user_agent="Sentiment Analysis Bot 1.0")

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    

    all_data = []
    for company in company_list:
        
        print(company)
        
        reddit_data=[]
        youtube_data=[]
        tumblr_data=[]
        google_news_data=[]
        ft_data=[]
        bloomberg_data=[]
        reuters_data=[]
        wsj_data=[]
        serper_data=[]

        # Social media data collection
        # twitter_data = collect_twitter_data(twitter_api, company, start_date, end_date)
        print("Collecting Reddit Data")
        reddit_data = collect_reddit_data(reddit, company, start_date, end_date)
        print("Collecting Youtube Data")
        youtube_data = collect_youtube_data(youtube, company, start_date, end_date)
        print("Collecting Tumblr Data")
        tumblr_data = collect_tumblr_data(tumblr_client, company, start_date, end_date)
        
        # News data collection
        print("Collecting Google News Data")
        google_news_data = collect_google_news(company, start_date, end_date)
        print("Collecting Financial Times Data")
        ft_data = collect_financial_times(company, start_date, end_date)
        print("Collecting Bloomberg Data")
        bloomberg_data = collect_bloomberg(company, start_date, end_date)
        print("Collecting Reuters Data")
        reuters_data = collect_reuters(company, start_date, end_date)
        print("Collecting WSJ Data")
        wsj_data = collect_wsj(company, start_date, end_date)
        print("Collecting from Serper")
        serper_data = search_news(company)
        
        # all_data.extend([
            # {"platform": "Twitter", "company": company, "data": item} for item in twitter_data
        # ])
        all_data.extend([
            {"platform": "Reddit", "company": company, 
             "page_content": {"title":item["title"],
                              "content":item["content"]}} for item in reddit_data
        ])
        all_data.extend([
            {"platform": "YouTube", "company": company, 
              "page_content": {"title":item["snippet"]["title"],
                               "content":item["snippet"]["description"]}} for item in youtube_data
        ])
        all_data.extend([
            {"platform": "Tumblr", "company": company, 
             "page_content": {"title":item["blog"]["title"],
                               "content":item["blog"]["description"]}} for item in tumblr_data
        ])
        
        all_data.extend([
            {"platform": "Google News", "company": company, 
             "page_content": {"title":item["title"],
                               "content":item["description"]}} for item in google_news_data
        ])
        all_data.extend([
            {"platform": "Financial Times", "company": company, 
              "page_content": {"title":item["title"],
                               "content":item["description"]}} for item in ft_data
        ])
        all_data.extend([
            {"platform": "Bloomberg", "company": company, 
              "page_content": {"title":item["title"],
                               "content":item["description"]}} for item in bloomberg_data
        ])
        all_data.extend([
            {"platform": "Reuters", "company": company, 
              "page_content": {"title":item["title"],
                               "content":item["description"]}} for item in reuters_data
        ])
        all_data.extend([
            {"platform": "Wall Street Journal", "company": company, 
              "page_content": {"title":item["title"],
                               "content":item["description"]}} for item in wsj_data
        ])
        all_data.extend([
            {"platform": item["source"], "company": company, 
              "page_content": {"title":item["title"],
                               "content":item["snippet"]}} for item in serper_data
        ])
    
    return all_data




    
# social_media_data = collect_social_media_and_news_data(company_list, start_date, end_date)
    
#     # Next steps: Send data to ingestion layer

# print(social_media_data)





