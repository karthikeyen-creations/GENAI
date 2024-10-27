import sys
import json
import os
import requests
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
import pandas as pd
from datetime import datetime
from dotted_dict import DottedDict
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from py.data_fetch import DataFetch
from py.handle_files import *
from py.db_storage import DBStorage
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class StockAdviserConfig:
    def __init__(self):
        load_dotenv()
        self.azure_config = {
            "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "embedding_base_url": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            "model_deployment": os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
            "model_name": os.getenv("AZURE_OPENAI_MODEL_NAME"),
            "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            "embedding_name": os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
        }
        self.models = DottedDict()

class StockAdviserUI:
    def __init__(self):
        st.set_page_config(page_title="AI Stock Adviser", layout="wide")
        self._setup_css()
        self._setup_header()

    def _setup_css(self):
        st.markdown("""
            <style>
            .main-header {
                text-align: center;
                padding-right: 20px;
                padding-left: 20px;
                color: #E9EBED;
            }
            .main-header2 {
                text-align: left;
                color: #E9EBED;
            }
            .column-header {
                color: #FFFF9E;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .column-header2 {
                color: #CEFFFF;
                padding-top: 5px;
                padding-bottom: 5px;
            }
            .content-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

    def _setup_header(self):
        st.markdown("<h1 class='main-header'>Karthikeyen Assignment - RAG Stock Analysis</h1>", unsafe_allow_html=True)
        st.markdown(""" <h4 class='main-header2'>
            Please ask if now is a good time to buy or sell NSE stocks of a company of your interest. 
            \n \n
            Note: For Demo purpose, historical data is available only for the below companies:   
            Reliance Industries,RELIANCE
            HDFC Bank,HDFCBANK
            Hindustan Unilever,HINDUNILVR
            Bharti Airtel,BHARTIARTL
            Asian Paints,ASIANPAINT
            Maruti Suzuki India,MARUTI
            """, 
            unsafe_allow_html=True)

class StockAdviser:
    def __init__(self):
        self.config = StockAdviserConfig()
        self.ui = StockAdviserUI()
        self.client = AzureOpenAI(
            azure_endpoint=self.config.azure_config["base_url"],
            api_key=self.config.azure_config["api-key"],
            api_version="2024-02-01"
        )

    def create_models(self):
        print("creating models")
        llm = AzureChatOpenAI(
            temperature=0,
            api_key=self.config.azure_config["api-key"],
            openai_api_version=self.config.azure_config["api_version"],
            azure_endpoint=self.config.azure_config["base_url"],
            model=self.config.azure_config["model_deployment"],
            validate_base_url=False
        )
        embedding_model = AzureOpenAIEmbeddings(
            api_key=self.config.azure_config["api-key"],
            openai_api_version=self.config.azure_config["api_version"],
            azure_endpoint=self.config.azure_config["embedding_base_url"],
            model=self.config.azure_config["embedding_deployment"]
        )
        self.config.models.llm = llm
        self.config.models.embedding_model = embedding_model
        return self.config.models

    def get_symbol(self, user_question):
        qna_system_message = """
        You are an assistant to a financial services firm who finds the 'nse company symbol' (assigned to the company in the provided stock market)) of the company in the question provided.

        User questions will begin with the token: ###Question.

        Please find the 'nse company symbol' of the company in the question provided. In case of an invalid company, return "NOTICKER".
        
        Response format:
        {nse company symbol}
        
        Do not mention anything about the context in your final answer. Stricktly respond only the company symbol.
        """

        qna_user_message_template = """
        ###Question
        {question}
        """

        prompt = [
            {'role': 'system', 'content': qna_system_message},
            {'role': 'user', 'content': qna_user_message_template.format(question=user_question)}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.azure_config["model_name"],
                messages=prompt,
                temperature=0
            )
            cmp_tkr = response.choices[0].message.content.strip()
        except Exception as e:
            cmp_tkr = f'Sorry, I encountered the following error: \n {e}'
            st.write("Reply: ", cmp_tkr)
            return
        print(cmp_tkr)
        return cmp_tkr

    def process_historical_data(self, user_question):
        cmp_tr = self.get_symbol(user_question)
        
        # Initialize ChromaDB Database
        chroma_db = DBStorage()
        FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_HD")
        chroma_db.load_vectors(FAISS_DB_PATH)
        context_for_query = chroma_db.get_context_for_query(cmp_tr, k=5)
        
        sentiment_response = self._get_sentiment_analysis(context_for_query, cmp_tr)
        self._display_sentiment(sentiment_response)
        return cmp_tr

    def process_realtime_data(self, cmp_tr):
        if cmp_tr == "NOTICKER":
            st.write("No valid company in the query.")
            return

        data_fetch = DataFetch()
        query_context = []
        
        # Create a placeholder for the current source
        source_status = st.empty()
        
        # Collect data from various sources
        data_sources = [
            ("Reddit", data_fetch.collect_reddit_data),
            ("YouTube", data_fetch.collect_youtube_data),
            ("Tumblr", data_fetch.collect_tumblr_data),
            ("Google News", data_fetch.collect_google_news),
            ("Financial Times", data_fetch.collect_financial_times),
            ("Bloomberg", data_fetch.collect_bloomberg),
            ("Reuters", data_fetch.collect_reuters)
        ]
        
        st_status = ""

        for source_name, collect_func in data_sources:
            st_status = st_status.replace("Currently fetching", "Fetched") + f"📡 Currently fetching data from: {source_name} \n \n"
            source_status.write(st_status, unsafe_allow_html=True)
            print(f"Collecting {source_name} Data")
            query_context.extend(collect_func(cmp_tr))

        st_status = st_status.replace("Currently fetching", "Fetched") +  "📡 Currently fetching data from: Serper - StockNews, Yahoo Finance, Insider Monkey, Investor's Business Daily, etc."
        source_status.write(st_status, unsafe_allow_html=True)
        print("Collecting Serper Data")
        query_context.extend(data_fetch.search_news(cmp_tr, 100))

        # Process collected data
        db_store = DBStorage()
        FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_RD")
        db_store.embed_vectors(to_documents(query_context), FAISS_DB_PATH)
        
        db_store.load_vectors(FAISS_DB_PATH)
        context_for_query = db_store.get_context_for_query(cmp_tr, k=5)
        
        sentiment_response = self._get_sentiment_analysis(context_for_query, cmp_tr, is_realtime=True)
        self._display_sentiment(sentiment_response)
        
        # Clear the status message after all sources are processed
        source_status.empty()


    def _get_sentiment_analysis(self, context, cmp_tr, is_realtime=False):
        system_message = self._get_system_prompt(is_realtime)
        user_message = f"""
        ###Context
        Here are some documents that are relevant to the question mentioned below.
        {context}

        ###Question
        {cmp_tr}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.config.azure_config["model_name"],
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': user_message}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f'Sorry, I encountered the following error: \n {e}'

    def _display_sentiment(self, prediction):
        sentiment = self._extract_between(prediction, "Overall Sentiment:", "Overall Justification:").strip()
        print("Sentiment: "+ sentiment)
        print(prediction)
        if sentiment == "Positive":
            st.success("Positive : Go Ahead...!")
        elif sentiment == "Negative":
            st.warning("Negative : Don't...!")
        elif sentiment == "Neutral":
            st.info("Neutral : Need to Analyse further")
        st.write(prediction, unsafe_allow_html=True)

    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> str:
        try:
            start_pos = text.find(start)
            if start_pos == -1:
                return ""
            start_pos += len(start)
            end_pos = text.find(end, start_pos)
            if end_pos == -1:
                return ""
            return text[start_pos:end_pos]
        except (AttributeError, TypeError):
            return ""

    @staticmethod
    def _get_system_prompt(is_realtime):
        """
        Returns the appropriate system prompt based on whether it's realtime or historical data analysis.
        
        Args:
            is_realtime (bool): Flag indicating if this is for realtime data analysis
        
        Returns:
            str: The complete system prompt for the sentiment analysis
        """
        base_prompt = """
        You are an assistant to a financial services firm who answers user queries on Stock Investments.
        User input will have the context required by you to answer user questions.
        This context will begin with the token: ###Context.
        The context contains references to specific portions of a document relevant to the user query.

        User questions will begin with the token: ###Question.
        
        First, find the 'nse company symbol' of the related company in the question provided.
        Your task is to perform sentiment analysis on the content part of each documents provided in the Context, which discuss a company identified by its 'nse company symbol'. The goal is to determine the overall sentiment expressed across all documents and provide an overall justification. Based on the sentiment analysis, give a recommendation on whether the company's stock should be purchased.

        Step-by-Step Instructions:
            1. See if the question is "NOTICKER". If so, give response and don't proceed.
            2. If the company in question is not found in the context, give the corresponding response and don't proceed.
            3. Read the Context: Carefully read the content parts of each document provided in the list of Documents.
            4. Determine Overall Sentiment: Analyze the sentiment across all documents and categorize the overall sentiment as Positive, Negative, or Neutral.
            5. Provide Overall Justification: Summarize the key points from all documents to justify the overall sentiment.
            6. Stock Advice: Based on the overall sentiment and justification, provide a recommendation on whether the company's stock should be purchased.

        Example Analysis:
            Context: 
                [Document(metadata={'platform': 'Moneycontrol', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 134}, page_content="{'title': 'Asian Paints launches Neo Bharat Latex Paint to tap on booming demand', 'content': 'The company, which is the leading player in India, touts the new segment to being affordable, offering over 1000 shades for consumers.'}"), Document(metadata={'platform': 'MarketsMojo', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 128}, page_content="{'title': 'Asian Paints Ltd. Stock Performance Shows Positive Trend, Outperforms Sector by 0.9%', 'content': 'Asian Paints Ltd., a leading player in the paints industry, has seen a positive trend in its stock performance on July 10, 2024.'}"), Document(metadata={'platform': 'Business Standard', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 138}, page_content="{'title': 'Asian Paints, Indigo Paints, Kansai gain up to 5% on falling oil prices', 'content': 'Shares of paint companies were trading higher on Wednesday, rising up to 5 per cent on the BSE, on the back of a fall in crude oil prices.'}")]
        """

        if is_realtime:
            response_format = """
        Response Formats:
        Only If the Question is 'NOTICKER':
            No valid company in the query.
        
        Else, If the context does not have relevent data for the company:
            Respond "Company {Company name} {nse company symbol}({symbol}) details not found in the RealTime Data".
        """
        else:
            response_format = """
        Response Formats:
        If the Question value is "NOTICKER":
            No valid company in the query.
        
        If the context does not have relevent data for the company (Question value):
            Respond "Company {Company name} {nse company symbol}({symbol}) details not found in the Historical Data".
        """

        common_format = """
        else, If the content parts of context has relevent data:
        Overall Sentiment: [Positive/Negative/Neutral]  <line break>
        Overall Justification: [Detailed analysis of why the sentiment was chosen, summarizing key points from the documents]  <line break>
        Stock Advice: [Clear recommendation on whether to purchase the stock, based on the sentiment analysis and justification]
        
        Please follow the steps to analyze the sentiment of each document's content; and strictly follow exact structure illustrated in above example response to provide an overall sentiment, justification and give stock purchase advice. Provide only Overall response, don't provide documentwise response or any note. Decorate the response with html/css tags.
        """

        return base_prompt + response_format + common_format

def main():
    adviser = StockAdviser()
    st.header("Ask a question")
    user_question = st.text_input("Ask a stock advise related question", key="user_question")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 class='column-header'>From Historical Data</h2>", unsafe_allow_html=True)
        with st.container():
            if user_question:
                cmp_tr = adviser.process_historical_data(user_question)

    with col2:
        st.markdown("<h2 class='column-header'>From Real-Time Data</h2>", unsafe_allow_html=True)
        with st.container():
            if user_question:
                adviser.process_realtime_data(cmp_tr)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>© 2024 EY</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()