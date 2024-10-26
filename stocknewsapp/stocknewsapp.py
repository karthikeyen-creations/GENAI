import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, List
import json

# Configure environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "3a93fe43c4534148a7a193412e29a321"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ey-openai.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview"
os.environ["SERPER_API_KEY"] = "11487e106a0d86120555c2258b3a5e71bc09c9b1"


# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    api_version="2024-02-01",
    model_name="gpt-35-turbo",
    temperature=0.1
)

class NewsAnalyzer:
    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.news_cache = {}
    
    def search_news(self, query: str, days: int = 7) -> List[Dict]:
        """Search news using Serper API"""
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": f"{query} company news",
            "num": 10,
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

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using Azure OpenAI"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Analyze the sentiment of this news article. Return only POSITIVE, NEGATIVE, or NEUTRAL.\n\nText: {text}\n\nSentiment:"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)
        return response.strip()

    def analyze_market_impact(self, text: str, sentiment: str) -> str:
        """Analyze market impact using Azure OpenAI"""
        prompt = PromptTemplate(
            input_variables=["text", "sentiment"],
            template="Based on this news article and its {sentiment} sentiment, analyze the potential impact on market perception and stock movement. Be concise and specific.\n\nText: {text}\n\nMarket Impact:"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text, sentiment=sentiment)
        return response.strip()

    def summarize_article(self, text: str) -> List[str]:
        """Summarize article using Azure OpenAI"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following news article in 4-5 key bullet points:\n\nText: {text}\n\nKey Points:"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)
        return [point.strip() for point in response.split('\n') if point.strip()]

def main():
    st.set_page_config(page_title="Stock News Analyzer", layout="wide")
    st.title("Stock News Analyzer")
    
    # Initialize NewsAnalyzer
    analyzer = NewsAnalyzer()
    
    # User input
    company = st.text_input("Enter Company Name or Stock Symbol:")
    days = st.slider("Select news timeframe (days)", 1, 30, 7)
    
    if st.button("Analyze News"):
        if company:
            with st.spinner("Fetching and analyzing news..."):
                # Get news articles
                news_results = analyzer.search_news(company, days)
                
                if news_results:
                    for article in news_results[:5]:  # Analyze top 5 articles
                        st.subheader(article["title"])
                        
                        # Create columns for layout
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Display summary points
                            summary = analyzer.summarize_article(article["snippet"])
                            st.write("Key Points:")
                            for point in summary:
                                st.markdown(f"• {point}")
                            
                            # Display source and date
                            st.write(f"Source: [{article['source']}]({article['link']})")
                            st.write(f"Published: {article.get('date', 'Date not available')}")
                        
                        with col2:
                            # Analyze sentiment
                            sentiment = analyzer.analyze_sentiment(article["snippet"])
                            sentiment_color = {
                                "POSITIVE": "green",
                                "NEGATIVE": "red",
                                "NEUTRAL": "gray"
                            }
                            st.markdown(f"**Sentiment:** :{sentiment_color.get(sentiment, 'blue')}[{sentiment}]")
                            
                            # Analyze market impact
                            impact = analyzer.analyze_market_impact(article["snippet"], sentiment)
                            st.markdown("**Market Impact:**")
                            st.write(impact)
                        
                        st.divider()
                else:
                    st.error("No news articles found for the specified company.")
        else:
            st.warning("Please enter a company name or stock symbol.")

if __name__ == "__main__":
    main()