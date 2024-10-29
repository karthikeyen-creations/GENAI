import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import tweepy
import praw
import googleapiclient.discovery
import pytumblr
from gnews import GNews
import requests
from bs4 import BeautifulSoup
import time
import math


class DataFetch:
    def __init__(self):
        # Load company list and set date range
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=1)

        # Initialize API clients
        self.tumblr_client = pytumblr.TumblrRestClient(
            os.getenv("TUMBLR_CONSUMER_KEY"),
            os.getenv("TUMBLR_CONSUMER_SECRET"),
            os.getenv("TUMBLR_OAUTH_TOKEN"),
            os.getenv("TUMBLR_OAUTH_SECRET")
        )
        
        twitter_auth = tweepy.OAuthHandler(os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET"))
        twitter_auth.set_access_token(os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
        self.twitter_api = tweepy.API(twitter_auth)

        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="Sentiment Analysis Bot 1.0"
        )

        self.youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

    def load_company_list(self, file_path: str) -> List[str]:
        self.company_list = pd.read_csv(file_path)['company_ticker'].tolist()

    def collect_data(self) -> List[Dict]:
        all_data = []
        
        for company in self.company_list:
            print(f"{company}:")
            all_data.extend(self._collect_social_media_data(company))
            all_data.extend(self._collect_news_data(company))
        
        return all_data

    def _collect_social_media_data(self, query: str) -> List[Dict]:
        social_data = []

        print("Collecting Reddit Data")
        social_data.extend(self.collect_reddit_data(query))

        print("Collecting YouTube Data")
        social_data.extend(self.collect_youtube_data(query))

        print("Collecting Tumblr Data")
        social_data.extend(self.collect_tumblr_data(query))

        return social_data

    def _collect_news_data(self, query: str) -> List[Dict]:
        news_data = []

        print("Collecting Google News Data")
        news_data.extend(self.collect_google_news(query))

        print("Collecting Financial Times Data")
        news_data.extend(self.collect_financial_times(query))

        print("Collecting Bloomberg Data")
        news_data.extend(self.collect_bloomberg(query))

        print("Collecting Reuters Data")
        news_data.extend(self.collect_reuters(query))

        print("Collecting WSJ Data")
        # news_data.extend(self.collect_wsj(query))

        print("Collecting Serper Data - StockNews, Yahoo Finance, Insider Monkey, Investor's Business Daily, etc.")
        news_data.extend(self.search_news(query))

        return news_data

    def collect_tumblr_data(self, query: str) -> List[Dict]:
        posts = self.tumblr_client.tagged(query)
        return [{"platform": "Tumblr", "company": query, "page_content": {
            "title": post["blog"]["title"], "content": post["blog"]["description"]}} for post in posts]

    def collect_twitter_data(self, query: str) -> List[Dict]:
        tweets = []
        for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query, lang="en",
                                   since=self.start_date, until=self.end_date).items(100):
            tweets.append(tweet._json)
        return [{"platform": "Twitter", "company": query, "page_content": tweet} for tweet in tweets]

    def collect_reddit_data(self, query: str) -> List[Dict]:
        posts = []
        subreddit = self.reddit.subreddit("all")
        for post in subreddit.search(query, sort="new", time_filter="day"):
            post_date = datetime.fromtimestamp(post.created_utc)
            if self.start_date <= post_date <= self.end_date:
                posts.append({"platform": "Reddit", "company": query, "page_content": {
                    "title": post.title, "content": post.selftext}})
        return posts

    def collect_youtube_data(self, query: str) -> List[Dict]:
        request = self.youtube.search().list(
            q=query, type="video", part="id,snippet", maxResults=50,
            publishedAfter=self.start_date.isoformat() + "Z", publishedBefore=self.end_date.isoformat() + "Z"
        )
        response = request.execute()
        return [{"platform": "YouTube", "company": query, "page_content": {
            "title": item["snippet"]["title"], "content": item["snippet"]["description"]}} for item in response['items']]

    def collect_google_news(self, query: str) -> List[Dict]:
        google_news = GNews(language='en', country='US', start_date=self.start_date, end_date=self.end_date)
        articles = google_news.get_news(query)
        return [{"platform": "Google News", "company": query, "page_content": {
            "title": article["title"], "content": article["description"]}} for article in articles]

    def collect_financial_times(self, query: str) -> List[Dict]:
        url = f"https://www.ft.com/search?q={query}&dateTo={self.end_date.strftime('%Y-%m-%d')}&dateFrom={self.start_date.strftime('%Y-%m-%d')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('div', class_='o-teaser__content')
        return [{"platform": "Financial Times", "company": query, "page_content": {
            "title": a.find('div', class_='o-teaser__heading').text.strip(),
            "content": a.find('p', class_='o-teaser__standfirst').text.strip() if a.find('p', class_='o-teaser__standfirst') else ''
        }} for a in articles]

    def collect_bloomberg(self, query: str) -> List[Dict]:
        url = f"https://www.bloomberg.com/search?query={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('div', class_='storyItem__aaf871c1')
        return [{"platform": "Bloomberg", "company": query, "page_content": {
            "title": a.find('a', class_='headline__3a97424d').text.strip(),
            "content": a.find('p', class_='summary__483358e1').text.strip() if a.find('p', class_='summary__483358e1') else ''
        }} for a in articles]

    def collect_reuters(self, query: str) -> List[Dict]:
        articles = []
        base_url = "https://www.reuters.com/site-search/"
        page = 1
        while True:
            url = f"{base_url}?blob={query}&page={page}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            results = soup.find_all('li', class_='search-result__item')
            if not results:
                break
            for result in results:
                date_elem = result.find('time', class_='search-result__timestamp')
                if date_elem:
                    date = datetime.strptime(date_elem['datetime'], "%Y-%m-%dT%H:%M:%SZ")
                    if self.start_date <= date <= self.end_date:
                        articles.append({"platform": "Reuters", "company": query, "page_content": {
                            "title": result.find('h3', class_='search-result__headline').text.strip(),
                            "content": result.find('p', class_='search-result__excerpt').text.strip()
                        }})
                    elif date < self.start_date:
                        return articles
            page += 1
            time.sleep(1)
        return articles

    def collect_wsj(self, query: str) -> List[Dict]:
        articles = []
        base_url = "https://www.wsj.com/search"
        page = 1
        while True:
            params = {
                'query': query, 'isToggleOn': 'true', 'operator': 'AND', 'sort': 'date-desc',
                'duration': 'custom', 'startDate': self.start_date.strftime('%Y/%m/%d'),
                'endDate': self.end_date.strftime('%Y/%m/%d'), 'page': page
            }
            response = requests.get(base_url, params=params)
            soup = BeautifulSoup(response.content, 'html.parser')
            results = soup.find_all('article', class_='WSJTheme--story--XB4V2mLz')
            if not results:
                break
            for result in results:
                date_elem = result.find('p', class_='WSJTheme--timestamp--22sfkNDv')
                if date_elem:
                    date = datetime.strptime(date_elem.text.strip(), "%B %d, %Y")
                    if self.start_date <= date <= self.end_date:
                        articles.append({"platform": "Wall Street Journal", "company": query, "page_content": {
                            "title": result.find('h3', class_='WSJTheme--headline--unZqjb45').text.strip(),
                            "content": result.find('p', class_='WSJTheme--summary--lmOXEsbN').text.strip()
                        }})
                    elif date < self.start_date:
                        return articles
            page += 1
            time.sleep(1)
        return articles

    def search_news(self, query: str,cnt=300) -> List[Dict]:
        articles = []
        num_results = cnt

        headers = {
            "X-API-KEY": os.getenv("SERP_API_KEY"),
            "Content-Type": "application/json"
        }
        payload = {"q": f"{query} company news",
            "num": num_results,
            "dateRestrict": 14
            }
        response = requests.post(
            "https://google.serper.dev/news",
            headers=headers,
            json=payload
            )
        # print(response)
        if response.status_code == 200:
            results = response.json().get("news", [])
            for result in results:
                articles.append({"platform": result["source"], "company": query, "page_content": {
                    "title": result["title"],
                    "content": result["snippet"],
                    "link": result["link"]
                }})
        return articles

# Usage Example
if __name__ == "__main__":
    analyzer = DataFetch("company_list.csv")
    data = analyzer.collect_data()
    # Here, data would contain all collected sentiment data for the given companies
