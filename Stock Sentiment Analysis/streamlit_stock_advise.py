import sys
import json
import os
import requests
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from openai import AzureOpenAI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
import yfinance as yf

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
        st.set_page_config(page_title="GEN AI Stock Adviser by Karthikeyen", layout="wide",
                           initial_sidebar_state="expanded")
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
                # margin-bottom: 2rem;
            }
            .little-header {
                # text-align: center;
                # padding-right: 20px;
                # padding-left: 20px;
                color: #E9EBED;
                # margin-bottom: 2rem;
            }
            .main-header2 {
                text-align: left;
                color: #E9EBED;
            }
            .column-header {
                color: #FFFF9E;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 1.5rem;
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
            .metric-card {
                background-color: #1E1E1E;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            .metric-title {
                font-size: 0.9rem;
                color: #888;
                margin-bottom: 0.5rem;
            }
            .metric-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #fff;
            }
            </style>
            """, unsafe_allow_html=True)

    def _setup_header(self):
        st.markdown("<h1 class='main-header'>Stock Analysis with Generative AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 class='main-header'>using Agents and RAG</h3>", unsafe_allow_html=True)
        with st.expander("Available Historical Demo Companies"):
            st.markdown("""
                For Demo purpose, historical data is available only for the below companies:
                - Reliance Industries (RELIANCE)
                - HDFC Bank (HDFCBANK)
                - Hindustan Unilever (HINDUNILVR)
                - Bharti Airtel (BHARTIARTL)
                - Asian Paints (ASIANPAINT)
                - Maruti Suzuki India (MARUTI)
            """, unsafe_allow_html=True)

class StockDataVisualizer:
    @staticmethod
    def create_price_chart(df, symbol):
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ))
        
        fig.update_layout(
            title=f'{symbol} Stock Price Movement',
            yaxis_title='Stock Price (INR)',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig

    @staticmethod
    def create_volume_chart(df, symbol):
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0, 150, 255, 0.6)'
        ))
        
        fig.update_layout(
            title=f'{symbol} Trading Volume',
            yaxis_title='Volume',
            template='plotly_dark',
            height=300
        )
        
        return fig

    @staticmethod
    def create_sentiment_gauge(sentiment_score):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "rgba(0, 150, 255, 0.6)"},
                'steps': [
                    {'range': [-1, -0.25], 'color': "red"},
                    {'range': [-0.25, 0.25], 'color': "yellow"},
                    {'range': [0.25, 1], 'color': "green"}
                ]
            },
            title={'text': "Sentiment Score"}
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=250
        )
        
        return fig

class StockAdviser:
    def __init__(self):
        self.config = StockAdviserConfig()
        self.ui = StockAdviserUI()
        self.visualizer = StockDataVisualizer()
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

    def stock_agent(self, user_question):
        functions=[
                {
                    "name":"get_advise",
                    "description":"Get only advise on a NSE stock",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "company":{
                                "type":"string",
                                "description":"Please find the 'nse company symbol' of the company in the question provided. In case of an invalid company, return 'NOTICKER'.",
                            },
                            
                        },
                        "required":["company"]
                    },
                },
                {
                    "name":"get_stats",
                    "description":"Get only statistics/status on a NSE stock",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "company":{
                                "type":"string",
                                "description":"Please find the 'nse company symbol' of the company in the question provided. In case of an invalid company, return 'NOTICKER'.",
                            },
                            
                        },
                        "required":["company"]
                    },
                },
                {
                    "name":"get_adv_stats",
                    "description":"Get both advise and statistics/status on a NSE stock",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "company":{
                                "type":"string",
                                "description":"Please find the 'nse company symbol' of the company in the question provided. In case of an invalid company, return 'NOTICKER'.",
                            },
                            
                        },
                        "required":["company"]
                    },
                },
                {
                    "name":"get_none",
                    "description":"Get details other than advise or statistics/status on a NSE stock",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "company":{
                                "type":"string",
                                "description":"""
                                For any queries other than advise or statistics/status on a NSE stock, only return "NOTICKER".
                                """,
                            },
                            
                        },
                        "required":["company"]
                    },
                }
            ] 

        
        initial_response = self.client.chat.completions.create(
            model=self.config.azure_config["model_deployment"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant to understand the context of input query on NSE stock advise and statistics."},
                {"role": "user", "content": user_question}
            ],
        functions=functions
        )

        print (initial_response)
        function_name = initial_response.choices[0].message.function_call.name
        function_argument = json.loads(initial_response.choices[0].message.function_call.arguments)
        company= function_argument['company']
        print(function_name)
        print(company)
        return function_name

    
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


    def process_historical_data(self, cmp_tr, hugg = False):
        
        # Initialize ChromaDB Database
        chroma_db = DBStorage(hugg)
        FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_HD")
        if hugg:
            FAISS_DB_PATH = os.path.join(os.getcwd(), "faiss_HD")
        chroma_db.load_vectors(FAISS_DB_PATH)
        context_for_query = chroma_db.get_context_for_query(cmp_tr, k=5)
        
        sentiment_response = self._get_sentiment_analysis(context_for_query, cmp_tr)
        self._display_sentiment(sentiment_response)
        
        return cmp_tr
    
    def display_charts(self,cmp_tr,sentiment_response="none"):
        
        days = 365
        
        print(f"\nFetching {days} days of stock data for {cmp_tr}...")
        df, analysis = self.get_nse_stock_data(cmp_tr, days)
        
        print("df,analysis")
        print(len(df))
        print(len(analysis))
        
        if len(analysis) != 0:
            # Create metrics cards
            
            col0, col1, col2, col3 = st.columns(4)
            # Simulate some metric data (replace with real data in production)\
            with col0:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">current price & volume</div>
                        <div class="metric-value">₹{analysis['current_price']:,}</div>
                        <div>{int(analysis['current_volume']):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
            with col1:
                self._create_metric_card(f"52-Week High on {analysis['week_52_high_date']}", 
                                        f"₹{analysis['week_52_high']:,.2f}", 
                                        self.format_percentage(analysis['pct_from_52w_high']))
            with col2:
                self._create_metric_card(f"52-Week Low on {analysis['week_52_low_date']}",  
                                        f"₹{analysis['week_52_low']:,.2f}", 
                                        self.format_percentage(analysis['pct_from_52w_low']))
            with col3:
                self._create_metric_card("Average Volume", 
                                        f"{int(analysis['avg_volume']):,}", 
                                        f"{self.format_percentage(analysis['volume_pct_diff'])}")
                    
            # Display price chart
            st.plotly_chart(self.visualizer.create_price_chart(df, cmp_tr))
            
            # Display volume chart
            st.plotly_chart(self.visualizer.create_volume_chart(df, cmp_tr))
            
            if sentiment_response != "none":
                sentiment = self._extract_between(sentiment_response, "Overall Sentiment:", "Overall Justification:").strip()
            
                # Display sentiment gauge (simulate sentiment score)
                # Generating random score for Demo purpose
                if sentiment == "Negative":
                    sentiment_score = np.random.uniform(-1, -0.75)
                elif sentiment == "Neutral":
                    sentiment_score = np.random.uniform(-0.75, 0.25)
                elif sentiment == "Positive":
                    sentiment_score = np.random.uniform(0.25, 1)
                else:
                    sentiment_score = 0
                    
                st.plotly_chart(self.visualizer.create_sentiment_gauge(sentiment_score))
    
    def get_nse_stock_data(self,symbol, days):
        """
        Fetch stock data and perform extended analysis including 52-week highs/lows
        and volume comparisons.
        
        Args:
            symbol (str): NSE stock symbol (e.g., 'RELIANCE.NS')
        
        Returns:
            tuple: (DataFrame of daily data, dict of analysis metrics)
        """
        try:
            # Add .NS suffix if not present
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Create Ticker object and fetch 1 year of data
            ticker = yf.Ticker(symbol)
            
            # Get last 90 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df_90d = ticker.history(start=start_date, end=end_date)
            
            # Get 1 year of data for 52-week analysis
            start_date_52w = end_date - timedelta(days=365)
            df_52w = ticker.history(start=start_date_52w, end=end_date)
            
            # Create main DataFrame with 90-day data
            df = pd.DataFrame({
                'Open': df_90d['Open'],
                'High': df_90d['High'],
                'Low': df_90d['Low'],
                'Close': df_90d['Close'],
                'Volume': df_90d['Volume']
            }, index=df_90d.index)
            
            # Round numerical values
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
            df['Volume'] = df['Volume'].astype(int)
            
            # Get current price (latest close)
            current_price = df['Close'].iloc[-1]
            
            # Calculate 52-week metrics
            week_52_high = df_52w['High'].max()
            week_52_low = df_52w['Low'].min()
            
            # Calculate percentage differences
            pct_from_52w_high = ((current_price - week_52_high) / week_52_high) * 100
            pct_from_52w_low = ((current_price - week_52_low) / week_52_low) * 100
            
            # Volume analysis
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df_52w['Volume'].mean()
            volume_pct_diff = ((current_volume - avg_volume) / avg_volume) * 100
                    
            # Find dates of 52-week high and low
            high_date = df_52w[df_52w['High'] == week_52_high].index[0].strftime('%Y-%m-%d')
            low_date = df_52w[df_52w['Low'] == week_52_low].index[0].strftime('%Y-%m-%d')
        
            # Create analysis metrics dictionary
            analysis = {
                'current_price': current_price,
                'week_52_high': week_52_high,
                'week_52_high_date': high_date,
                'week_52_low': week_52_low,
                'week_52_low_date': low_date,
                'pct_from_52w_high': pct_from_52w_high,
                'pct_from_52w_low': pct_from_52w_low,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_pct_diff': volume_pct_diff
            }
            
            print(analysis)
            
            return df, analysis
        
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return [], []

    def format_percentage(self, value):
        """Format percentage with + or - sign"""
        return f"+{value:.2f}%" if value > 0 else f"{value:.2f}%"


    def process_realtime_data(self, cmp_tr, hugg = False):
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
        db_store = DBStorage(hugg)
        FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_RD")
        
        if hugg:
            FAISS_DB_PATH = os.path.join(os.getcwd(), "faiss_RD")
            
        db_store.embed_vectors(to_documents(query_context), FAISS_DB_PATH)
        
        db_store.load_vectors(FAISS_DB_PATH)
        context_for_query = db_store.get_context_for_query(cmp_tr, k=5)
        
        sentiment_response = self._get_sentiment_analysis(context_for_query, cmp_tr, is_realtime=True)
        self._display_sentiment(sentiment_response)
        
        # Clear the status message after all sources are processed
        source_status.empty()
        
        return sentiment_response


    def _create_metric_card(self, title, value, change):
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div style="color: {'green' if float(change.strip('%')) > 0 else 'red'}">
                    {change}
                </div>
            </div>
        """, unsafe_allow_html=True)

    def _get_sentiment_analysis(self, context, cmp_tr, is_realtime=False):
        system_message, dcument = self._get_system_prompt(is_realtime)
        user_message = f"""
        ###Context
        Here are some list of {dcument} that are relevant to the question mentioned below.
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
            return f'Sorry, I encountered the following error: \n {e}', ""

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
        
        if is_realtime:
            response_format = """
                Response Formats:
                Only If the Question is 'NOTICKER':
                    No valid company in the query.
                
                Else, If the context does not have relevent data for the company:
                    Respond "Company {Company name} {nse company symbol}({symbol}) details not found in the RealTime Data".
                """
            citation_format = """
                Citations: [Generate few citations based on the links provided. Mention Source ('platform') and Title('title'), linking them with url from corresponding 'link' ]
                """
            instr2 = """
            Stricktly Never mention 'document'/'documents', not even once. Instead mention it as 'Real-Time Social media and News data'
            """
            
            dcument = "Real-Time Social media and News data"
            dcuments = "List of 'Real-Time Social media and News data'"
        else:
            response_format = """
                Response Formats:
                If the Question value is "NOTICKER":
                    No valid company in the query.
                
                If the context does not have relevent data for the company (Question value):
                    Respond "Company {Company name} {nse company symbol}({symbol}) details not found in the Historical Data".
                """
            citation_format = ""
                        
            instr2 = """
            Never mention 'document'/'documents', not even once. Instead mention it as 'Historical Social media and News data'
            """
            
            dcument = "Historical Social media and News data"
            dcuments = "List of 'Historical Social media and News data'"

        instr = f"""
        Please follow the steps to analyze the sentiment of each {dcument}'s content; and strictly follow exact structure illustrated in above example response to provide an overall sentiment, justification and give stock purchase advice. Provide only Overall response, don't provide documentwise response or any note. Decorate the response with html/css tags.
        """
        common_format = f"""
        else, If the content parts of context has relevent data:
        Overall Sentiment: [Positive/Negative/Neutral]  <line break>
        Overall Justification: [Detailed analysis of why the sentiment was chosen, summarizing key points from the {dcuments}]  <line break>
        Stock Advice: [Clear recommendation on whether to purchase the stock, based on the sentiment analysis and justification]
        """
        
        base_prompt = f"""
        You are an assistant to a financial services firm who answers user queries on Stock Investments.
        User input will have the context required by you to answer user questions.
        This context will begin with the token: ###Context.
        The context contains references to specific portions of a {dcument} relevant to the user query.
        Each document is a {dcument}.

        User questions will begin with the token: ###Question.
        
        First, find the 'nse company symbol' of the related company in the question provided.
        Your task is to perform sentiment analysis on the content part of each {dcuments} provided in the Context, which discuss a company identified by its 'nse company symbol'. The goal is to determine the overall sentiment expressed across all {dcuments} and provide an overall justification. Based on the sentiment analysis, give a recommendation on whether the company's stock should be purchased.

        Step-by-Step Instructions:
            1. See if the question is "NOTICKER". If so, give response and don't proceed.
            2. If the company in question is not found in the context, give the corresponding response and don't proceed.
            3. Read the Context: Carefully read the content parts of each {dcument} provided in the list of {dcuments}.
            4. Determine Overall Sentiment: Analyze the sentiment across all {dcuments} and categorize the overall sentiment as Positive, Negative, or Neutral.
            5. Provide Overall Justification: Summarize the key points from all {dcuments} to justify the overall sentiment.
            6. Stock Advice: Based on the overall sentiment and justification, provide a recommendation on whether the company's stock should be purchased.
        """
        example_analysis = """
        Example Analysis:
            Context: 
                [Document(metadata={'platform': 'Moneycontrol', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 134}, page_content="{'title': 'Asian Paints launches Neo Bharat Latex Paint to tap on booming demand', 'content': 'The company, which is the leading player in India, touts the new segment to being affordable, offering over 1000 shades for consumers.'}"), Document(metadata={'platform': 'MarketsMojo', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 128}, page_content="{'title': 'Asian Paints Ltd. Stock Performance Shows Positive Trend, Outperforms Sector by 0.9%', 'content': 'Asian Paints Ltd., a leading player in the paints industry, has seen a positive trend in its stock performance on July 10, 2024.'}"), Document(metadata={'platform': 'Business Standard', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 138}, page_content="{'title': 'Asian Paints, Indigo Paints, Kansai gain up to 5% on falling oil prices', 'content': 'Shares of paint companies were trading higher on Wednesday, rising up to 5 per cent on the BSE, on the back of a fall in crude oil prices.'}")]
        """


        return base_prompt + example_analysis + response_format + common_format + citation_format + instr + instr2, dcument

def get_advise(user_question,adviser,cmp_tr,sentiment_response,hugg):
    col1, col2 = st.columns(2)
    with col1:
        if user_question:
            st.markdown("<h3 class='little-header'>Historical Analysis</h3>", unsafe_allow_html=True)
            with st.container():
                adviser.process_historical_data(cmp_tr, hugg)

    with col2:
        if user_question:
            st.markdown("<h3 class='little-header'>Real-Time Analysis</h3>", unsafe_allow_html=True)
            with st.container():
                sentiment_response = adviser.process_realtime_data(cmp_tr, hugg)
                
    return sentiment_response
 
def get_stats(user_question,adviser,cmp_tr,sentiment_response,hugg):
    if (str(cmp_tr) != "NOTICKER"):            
        with st.container():
            if user_question:
                adviser.display_charts(cmp_tr,sentiment_response)

def get_adv_stats(user_question,adviser,cmp_tr,sentiment_response,hugg):
    sentiment_response = get_advise(user_question,adviser,cmp_tr,sentiment_response,hugg)     
    get_stats(user_question,adviser,cmp_tr,sentiment_response,hugg)

def get_none(user_question,adviser,cmp_tr,sentiment_response,hugg):
    st.write("Please enter a valid NSE stock enquiry.")

def main(hugg):
    adviser = StockAdviser()
    

    # Create sidebar for filters and settings
    st.logo(
    "https://cdn.shopify.com/s/files/1/0153/8513/3156/files/info_omac.png?v=1595717396",
    size="large"
    )

    with st.sidebar:
        # About the Application (Main Area)
        st.markdown("""
            <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);">
                <h2 style="color: #e6e6e6; text-align: center;">About the Application</h2>
                <p style="font-size: 16px; color: #d9d9d9; line-height: 1.6; text-align: justify;">
                    This application provides <span style="color: #80b1c1;"><strong>investment managers</strong></span> with daily insights into 
                    <span style="color: #d3b673;"><strong>social media</strong></span> and <span style="color: #d3b673;"><strong>news sentiment</strong></span> surrounding 
                    specific <span style="color: #80b1c1;"><strong>stocks and companies</strong></span>. By analyzing posts and articles across major platforms 
                    such as <strong style="color: #b0b0b0;">Reddit</strong>, <strong style="color: #b0b0b0;">YouTube</strong>, <strong style="color: #b0b0b0;">Tumblr</strong>, 
                    <strong style="color: #b0b0b0;">Google News</strong>, <strong style="color: #b0b0b0;">Financial Times</strong>, <strong style="color: #b0b0b0;">Bloomberg</strong>, 
                    <strong style="color: #b0b0b0;">Reuters</strong>, and <strong style="color: #b0b0b0;">Wall Street Journal</strong> (WSJ), it detects shifts in public 
                    and media opinion that may impact stock performance.
                </p>
                <p style="font-size: 16px; color: #d9d9d9; line-height: 1.6; text-align: justify;">
                    Additionally, sources like <span style="color: #80b1c1;"><strong>Serper</strong></span> provide data from 
                    <span style="color: #d3b673;"><strong>StockNews</strong></span>, <span style="color: #d3b673;"><strong>Yahoo Finance</strong></span>, 
                    <span style="color: #d3b673;"><strong>Insider Monkey</strong></span>, <span style="color: #d3b673;"><strong>Investor's Business Daily</strong></span>, 
                    and others. Using advanced <span style="color: #80b1c1;"><strong>AI techniques</strong></span>, the application generates a 
                    <span style="color: #d3b673;"><strong>sentiment report</strong></span> that serves as a leading indicator, helping managers make informed, 
                    timely adjustments to their positions. With daily updates and <span style="color: #d3b673;"><strong>historical trend analysis</strong></span>, 
                    it empowers users to stay ahead in a fast-paced, sentiment-driven market.
                </p>
                <p style="font-size: 16px; color: #d9d9d9; line-height: 1.6; text-align: justify;">
                    The application also utilizes <span style="color: #80b1c1;"><strong>intelligent agent functions</strong></span> to determine the type of query input 
                    by the user. It assesses whether the query seeks <span style="color: #d3b673;"><strong>stock statistics</strong></span>, 
                    <span style="color: #d3b673;"><strong>sentiment-analyzed advice</strong></span>, both, or is unrelated, providing the most relevant response accordingly.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar Footer (Floating Footer)
        st.sidebar.markdown("""
            <div style="position: fixed; bottom: 25px; background-color: #1f1f1f; padding: 1px; border-radius: 15px; text-align: center;">
                <p style="color: #cccccc; font-size: 14px; text-align: center; margin: 0;">
                    Developed by: <a href="https://www.linkedin.com/in/karthikeyen92/" target="_blank" style="color: #4DA8DA; text-decoration: none;">Karthikeyen Packirisamy</a>
                </p>
            </div>
        """, unsafe_allow_html=True)

                    

    # Main content
    cmp_tr = "NOTICKER"
    st.header("Ask a question")
    user_question = st.text_input("Please ask statistical or advice or both related questions on a NSE stock.", key="user_question")

    if user_question.strip():
        cmp_tr = adviser.get_symbol(user_question)
        sentiment_response = "none"
        
        agent_function = adviser.stock_agent(user_question)
        getattr(sys.modules[__name__], agent_function)(user_question,adviser,cmp_tr,sentiment_response,hugg)
        
    # get_adv_stats(user_question,adviser,cmp_tr,sentiment_response,hugg)
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>© 2024 Karthikeyen</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    hugg = os.getenv("IS_HUGG") == "True"
    print(hugg)
    main(hugg)