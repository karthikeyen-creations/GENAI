import sys
import json
import os
import requests
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
import pandas as pd
import os
from datetime import datetime
from dotted_dict import DottedDict
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from py.data_fetch import *
from py.handle_files import *


from langchain.callbacks import get_openai_callback
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

from py.db_storage import *

load_dotenv()

progress = ""
cmp_tr=""
stock_market="nse"

azure_config = {
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "embedding_base_url": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        "model_deployment": os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
        "model_name": os.getenv("AZURE_OPENAI_MODEL_NAME"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        "embedding_name": os.getenv("AZURE_OPENAI_EMBEDDING_NAME"),
        "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
}

models=DottedDict()

st.set_page_config(page_title="AI Stock Adviser", layout="wide")
# Add custom CSS to improve the layout
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

# Main title with custom styling
st.markdown("<h1 class='main-header'>Karthikeyen Assignment - RAG Stock Analysis</h1>", unsafe_allow_html=True)
st.markdown(""" <h4 class='main-header2'>
Please ask if now is a good time to buy or sell stocks of a company of your interest. 
\n \n
Note: For Demo purpose, historical data is available only for the below companies:   
Reliance Industries,RELIANCE
HDFC Bank,HDFCBANK
Hindustan Unilever,HINDUNILVR
Bharti Airtel,BHARTIARTL
Asian Paints,ASIANPAINT
Maruti Suzuki India,MARUTI

</h4>""", 
unsafe_allow_html=True)

def create_models(azure_config):
    print("creating models")
    llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["api-key"],
                      openai_api_version=azure_config["api_version"],
                      azure_endpoint=azure_config["base_url"],
                      model=azure_config["model_deployment"],
                      validate_base_url=False)
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_config["api-key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["embedding_base_url"],
        model = azure_config["embedding_deployment"]
    )
    models.llm=llm
    models.embedding_model=embedding_model 
    return models

def get_symbol(user_question):
    
    qna_system_message = """
    You are an assistant to a financial services firm who finds the 'nse company symbol' (assigned to the company in the provided stock market)) of the company in the question provided.

    User questions will begin with the token: ###Question.

    Please find the 'nse company symbol' of the company in the question provided. In case of an invalid company, return "NOTICKER".
    
    Response format:
    {nse company symbol}
    
    Do not mention anything about the context in your final answer.

    """

    qna_user_message_template = """
    ###Question
    {question}
    
    """

    client = AzureOpenAI(
    azure_endpoint=azure_config["base_url"],
    api_key=azure_config["api-key"],
    api_version="2024-02-01"
    )
    
    prompt = [
        {'role':'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            question=user_question
            )
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=azure_config["model_name"],
            messages=prompt,
            temperature=0
        )

        cmp_tkr = response.choices[0].message.content.strip()
    except Exception as e:
        cmp_tkr = f'Sorry, I encountered the following error: \n {e}'
        st.write("Reply: ", cmp_tkr)
        return
    print(cmp_tkr)
    return(cmp_tkr)
    
    
def user_input(user_question):
    print("Question: ", user_question)
    
    cmp_tr=get_symbol(user_question)
    
    # Initialise ChromaDB Database
    chroma_db = DBStorage()
    FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_HD")
    chroma_db.load_vectors(FAISS_DB_PATH)
    context_for_query = chroma_db.get_context_for_query(cmp_tr,k=5)
    
    
    qna_system_message = """
    You are an assistant to a financial services firm who answers user queries on Stock Investments.
    User input will have the context required by you to answer user questions.
    This context will begin with the token: ###Context.
    The context contains references to specific portions of a document relevant to the user query.

    User questions will begin with the token: ###Question.
    
    First, find the 'nse company symbol' of the related company in the question provided.
    Your task is to perform sentiment analysis on the content part of each documents provided in the Context, which discuss a company identified by its 'nse company symbol'. The goal is to determine the overall sentiment expressed across all documents and provide an overall justification. Based on the sentiment analysis, give a recommendation on whether the company’s stock should be purchased.

    Step-by-Step Instructions:
        1. See if the question is "NOTICKER". If so, give response and don't proceed.
        2. If the company in question is not found in the context, give the corresponding response and don't proceed.
        3. Read the Context: Carefully read the content parts of each document provided in the list of Documents.
        4. Determine Overall Sentiment: Analyze the sentiment across all documents and categorize the overall sentiment as Positive, Negative, or Neutral.
        5. Provide Overall Justification: Summarize the key points from all documents to justify the overall sentiment.
        6. Stock Advice: Based on the overall sentiment and justification, provide a recommendation on whether the company’s stock should be purchased.

    Example Analysis:
        Context: 
            [Document(metadata={'platform': 'Moneycontrol', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 134}, page_content="{'title': 'Asian Paints launches Neo Bharat Latex Paint to tap on booming demand', 'content': 'The company, which is the leading player in India, touts the new segment to being affordable, offering over 1000 shades for consumers.'}"), Document(metadata={'platform': 'MarketsMojo', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 128}, page_content="{'title': 'Asian Paints Ltd. Stock Performance Shows Positive Trend, Outperforms Sector by 0.9%', 'content': 'Asian Paints Ltd., a leading player in the paints industry, has seen a positive trend in its stock performance on July 10, 2024.'}"), Document(metadata={'platform': 'Business Standard', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 138}, page_content="{'title': 'Asian Paints, Indigo Paints, Kansai gain up to 5% on falling oil prices', 'content': 'Shares of paint companies were trading higher on Wednesday, rising up to 5 per cent on the BSE, on the back of a fall in crude oil prices.'}")]
    
    Response Formats:
    If the Question value is "NOTICKER":
        No valid company in the query.
    
    If the context does not have relevent data for the company (Question value):
        Respond "Company {Company name}Asian Paints {nse company symbol}(ASIANPAINT) details not found in the Historical Data".
    
    else, If the content parts of context has relevent data:
    Overall Sentiment: Positive <line break>
    Overall Justification: The documents collectively portray Asian Paints (ASIANPAINT) in a positive light. The company has launched an affordable product with over 1,000 shades, targeting a growing demand segment, which highlights its proactive approach to market trends. Additionally, Asian Paints' stock performance has shown a positive trend, outperforming its sector. Furthermore, a decline in crude oil prices has benefited the paint industry, boosting stock prices for key players, including Asian Paints. This consistent positive performance across various indicators suggests a stable outlook for the company. <line break>
    Stock Advice: Based on the favorable sentiment and strong market position, it is advisable to consider purchasing Asian Paints (ASIANPAINT) stock. The company’s market responsiveness, coupled with industry tailwinds, indicates potential for continued growth.
    
    Please follow the steps to analyze the sentiment of each document’s content; and stricktly follow exact structure illustrated in above example response to provide an overall sentiment, justification and give stock purchase advice. Provide only Overall response, don't provide documentwise response or any note. Decorate the reponse with html/css tags.
    
    """

    qna_user_message_template = """
    ###Context
    Here are some documents that are relevant to the question mentioned below.
    {context}

    ###Question
    {question}
    """

    client = AzureOpenAI(
    azure_endpoint=azure_config["base_url"],
    api_key=azure_config["api-key"],
    api_version="2024-02-01"
    )
    
    prompt = [
        {'role':'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            # question=user_question
            question=cmp_tr
            )
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=azure_config["model_name"],
            messages=prompt,
            temperature=0
        )

        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        prediction = f'Sorry, I encountered the following error: \n {e}'

    
    print(prediction)
    sentiment=extract_between(prediction,"Overall Sentiment: "," ").strip()
    if sentiment == "Positive":
        st.success("Positive : Go Ahead...!")
    elif sentiment == "Negative":
        st.warning("Negative : Don't...!")
    elif sentiment == "Neutral":
        st.info("Neutral : Need to Analyse further")
    st.write( prediction, unsafe_allow_html=True)
    return(cmp_tr)
    
    
def real_time(cmp_tr):
    
    # st.info("IN DEVELOPMENT")
    print(cmp_tr)
    if (cmp_tr == "NOTICKER"):
        st.write("No valid company in the query.")
    else:
        data_fetch = DataFetch()
        query_context = []
        print("Collecting Reddit Data")
        query_context.extend(data_fetch.collect_reddit_data(cmp_tr))

        print("Collecting YouTube Data")
        query_context.extend(data_fetch.collect_youtube_data(cmp_tr))

        print("Collecting Tumblr Data")
        query_context.extend(data_fetch.collect_tumblr_data(cmp_tr))
        

        print("Collecting Google News Data")
        query_context.extend(data_fetch.collect_google_news(cmp_tr))

        print("Collecting Financial Times Data")
        query_context.extend(data_fetch.collect_financial_times(cmp_tr))

        print("Collecting Bloomberg Data")
        query_context.extend(data_fetch.collect_bloomberg(cmp_tr))

        print("Collecting Reuters Data")
        query_context.extend(data_fetch.collect_reuters(cmp_tr))

        print("Collecting WSJ Data")
        # query_context.extend(data_fetch.collect_wsj(cmp_tr))

        print("Collecting Serper Data - StockNews, Yahoo Finance, Insider Monkey, Investor's Business Daily, etc.")
        query_context.extend(data_fetch.search_news(cmp_tr,100))

        print("collection done")
        print(len(query_context))
        
        db_store = DBStorage()
        FAISS_DB_PATH = os.path.join(os.getcwd(), "Stock Sentiment Analysis", "faiss_RD")
        db_store.embed_vectors(to_documents(query_context),FAISS_DB_PATH)
                
        # context_for_query = sample_documents(to_documents(query_context), 20)
        # context_for_query = query_vectors.get_context_for_query(cmp_tr,k=5)
        db_store.load_vectors(FAISS_DB_PATH)
        context_for_query = db_store.get_context_for_query(cmp_tr,k=5)
        
        print(context_for_query)
        
        qna_system_message = """
        You are an assistant to a financial services firm who answers user queries on Stock Investments.
        User input will have the context required by you to answer user questions.
        This context will begin with the token: ###Context.
        The context contains references to specific portions of a document relevant to the user query.

        User questions will begin with the token: ###Question.
        
        First, find the 'nse company symbol' of the related company in the question provided.
        Your task is to perform sentiment analysis on the content part of each documents provided in the Context, which discuss a company identified by its 'nse company symbol'. The goal is to determine the overall sentiment expressed across all documents and provide an overall justification. Based on the sentiment analysis, give a recommendation on whether the company’s stock should be purchased.

        Step-by-Step Instructions:
            1. See if the question is "NOTICKER". If so, give response and don't proceed.
            2. If the company in question is not found in the context, give the corresponding response and don't proceed.
            3. Read the Context: Carefully read the content parts of each document provided in the list of Documents.
            4. Determine Overall Sentiment: Analyze the sentiment across all documents and categorize the overall sentiment as Positive, Negative, or Neutral.
            5. Provide Overall Justification: Summarize the key points from all documents to justify the overall sentiment.
            6. Stock Advice: Based on the overall sentiment and justification, provide a recommendation on whether the company’s stock should be purchased.

        Example Analysis:
            Context: 
                [Document(metadata={'platform': 'Moneycontrol', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 134}, page_content="{'title': 'Asian Paints launches Neo Bharat Latex Paint to tap on booming demand', 'content': 'The company, which is the leading player in India, touts the new segment to being affordable, offering over 1000 shades for consumers.'}"), Document(metadata={'platform': 'MarketsMojo', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 128}, page_content="{'title': 'Asian Paints Ltd. Stock Performance Shows Positive Trend, Outperforms Sector by 0.9%', 'content': 'Asian Paints Ltd., a leading player in the paints industry, has seen a positive trend in its stock performance on July 10, 2024.'}"), Document(metadata={'platform': 'Business Standard', 'company': 'ASIANPAINT', 'ingestion_timestamp': '2024-10-25T17:13:42.970099', 'word_count': 138}, page_content="{'title': 'Asian Paints, Indigo Paints, Kansai gain up to 5% on falling oil prices', 'content': 'Shares of paint companies were trading higher on Wednesday, rising up to 5 per cent on the BSE, on the back of a fall in crude oil prices.'}")]
        
        Response Formats:
        Only If the Question is 'NOTICKER':
            No valid company in the query.
        
        Else, If the context does not have relevent data for the company:
            Respond "Company {Company name}Asian Paints {nse company symbol}(ASIANPAINT) details not found in the RealTime Data".
        
        else, If the content parts of context has relevent data:
        Overall Sentiment: Positive <line break>
        Overall Justification: The documents collectively portray Asian Paints (ASIANPAINT) in a positive light. The company has launched an affordable product with over 1,000 shades, targeting a growing demand segment, which highlights its proactive approach to market trends. Additionally, Asian Paints' stock performance has shown a positive trend, outperforming its sector. Furthermore, a decline in crude oil prices has benefited the paint industry, boosting stock prices for key players, including Asian Paints. This consistent positive performance across various indicators suggests a stable outlook for the company. <line break>
        Stock Advice: Based on the favorable sentiment and strong market position, it is advisable to consider purchasing Asian Paints (ASIANPAINT) stock. The company’s market responsiveness, coupled with industry tailwinds, indicates potential for continued growth.
        
        Please follow the steps to analyze the sentiment of each document’s content; and stricktly follow exact structure illustrated in above example response to provide an overall sentiment, justification and give stock purchase advice. Provide only Overall response, don't provide documentwise response or any note. Decorate the reponse with html/css tags.
        
        """

        qna_user_message_template = """

        ###Context
        Here are some documents that are relevant to the question mentioned below.
        {context}

        ###Question
        {question}
        """

        client = AzureOpenAI(
        azure_endpoint=azure_config["base_url"],
        api_key=azure_config["api-key"],
        api_version="2024-02-01"
        )
        
        prompt = [
            {'role':'system', 'content': qna_system_message},
            {'role': 'user', 'content': qna_user_message_template.format(
                context=context_for_query,
                question=cmp_tr
                )
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model=azure_config["model_name"],
                messages=prompt,
                temperature=0
            )

            prediction = response.choices[0].message.content.strip()
        except Exception as e:
            prediction = f'Sorry, I encountered the following error: \n {e}'

        
        print(prediction)
        sentiment=extract_between(prediction,"Overall Sentiment: "," ").strip()
        if sentiment == "Positive":
            st.success("Positive : Go Ahead...!")
        elif sentiment == "Negative":
            st.warning("Negative : Don't...!")
        elif sentiment == "Neutral":
            st.info("Neutral : Need to Analyse further")
        st.write( prediction, unsafe_allow_html=True)
    
def extract_between(text: str, start: str, end: str) -> str:
    """
    Extract substring between two delimiter strings.
    
    Args:
        text (str): The source text to extract from
        start (str): The starting delimiter string
        end (str): The ending delimiter string
    
    Returns:
        str: Extracted text between start and end delimiters
             Returns empty string if delimiters not found or invalid input
    
    Examples:
        >>> text = "Hello [world] Python"
        >>> extract_between(text, "[", "]")
        'world'
    """
    try:
        # Find starting position
        start_pos = text.find(start)
        if start_pos == -1:
            return ""
        
        # Adjust start position to end of start delimiter
        start_pos += len(start)
        
        # Find ending position after start position
        end_pos = text.find(end, start_pos)
        if end_pos == -1:
            return ""
        
        # Extract and return the text between delimiters
        return text[start_pos:end_pos]
        
    except (AttributeError, TypeError):
        return ""


def main():
    st.header("Ask a question")

    user_question = st.text_input("Ask a stock advise related question", key="user_question",)

    # Create two columns with equal width
    col1, col2 = st.columns(2)

    # First column content
    with col1:  
        st.markdown("<h2 class='column-header'>From Historical Data</h2>", unsafe_allow_html=True)
        
        with st.container():
            if user_question:
                cmp_tr = user_input(user_question)

    # Second column content
    with col2:
        st.markdown("<h2 class='column-header'>From Real-Time Data</h2>", unsafe_allow_html=True)
        
        with st.container():
            # progress = ""
            if user_question:
                # st.write(progress, unsafe_allow_html=True)
                real_time(cmp_tr)

    # Add a footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>© 2024 EY</p>", 
                unsafe_allow_html=True)
        # with st.sidebar:
        #     st.title("Menu:")
        #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        #     if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
        #         with st.spinner("Processing..."):
        #             raw_text = get_pdf_text(pdf_docs)
        #             text_chunks = get_text_chunks(raw_text)
        #             models=create_models(azure_config)
        #             get_vector_store(text_chunks, models)
        #             st.success("Done")
        # with st.form("my_form"):
        #     text = st.text_area("Enter text:", "How much is Vijay's experience?")
        #     submitted = st.form_submit_button("Submit")
        #     print(models)
        #     #create_models(azure_config)
        #     generate_response(text, models)
if __name__ == "__main__":
    main()
