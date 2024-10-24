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

cmp_tr=""

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
 AAPL  - Apple Inc.,                
 MSFT  - Microsoft Corporation,     
 AMZN  - Amazon.com Inc.,           
 GOOGL - Alphabet Inc. (Google)     

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

def get_ticker(user_question):
    
    qna_system_message = """
    You are an assistant to a financial services firm who finds the 'nasdaq company ticker' of the company in the question provided.

    User questions will begin with the token: ###Question.

    Please find the 'nasdaq company ticker' of the company in the question provided. In case of an invalid NASDAQ company, return "NOTICKER".
    
    Response format:
    {nasdaq company ticker}
    
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
    
    cmp_tr=get_ticker(user_question)
    
    # Initialise ChromaDB Database
    chroma_db = DBStorage()
    chroma_db.load_vectors()
    context_for_query = chroma_db.get_context_for_query(cmp_tr,k=5)
    
    
    qna_system_message = """
    You are an assistant to a financial services firm who answers user queries on Stock Investments.
    User input will have the context required by you to answer user questions.
    This context will begin with the token: ###Context.
    The context contains references to specific portions of a document relevant to the user query.

    User questions will begin with the token: ###Question.
    
    First, find the 'nasdaq company ticker' of the related company in the question provided.
    Your task is to perform sentiment analysis on the content part of each documents provided in the Context, which discuss a company identified by its 'nasdaq company ticker'. The goal is to determine the overall sentiment expressed across all documents and provide an overall justification. Based on the sentiment analysis, give a recommendation on whether the company’s stock should be purchased.

    Step-by-Step Instructions:
        1. Read the Context: Carefully read the content parts of each document provided in the list of Documents.
        2. Determine Overall Sentiment: Analyze the sentiment across all documents and categorize the overall sentiment as Positive, Negative, or Neutral.
        3. Provide Overall Justification: Summarize the key points from all documents to justify the overall sentiment.
        4. Stock Advice: Based on the overall sentiment and justification, provide a recommendation on whether the company’s stock should be purchased.

    Example Analysis:
        Context: 
            [Document(metadata={'platform': 'Google News', 'company': 'GOOGL', 'ingestion_timestamp': '2024-10-22T10:11:17.257275', 'word_count': 91}, page_content="{'title': 'Alphabet Inc. (GOOGL): Among the Most Owned Stocks by Hedge Funds Right Now - Insider Monkey', 'content': 'Alphabet Inc. (GOOGL): Among the Most Owned Stocks by Hedge Funds Right Now  Insider Monkey'}"),Document(metadata={'platform': 'Google News', 'company': 'GOOGL', 'ingestion_timestamp': '2024-10-22T10:11:17.257275', 'word_count': 59}, page_content="{'title': 'Here\'s Why Alphabet (GOOGL) is a Strong Momentum Stock - MSN', 'content': 'Here\'s Why Alphabet (GOOGL) is a Strong Momentum Stock  MSN'}")]
    
    Response Format:
    If the Question is "NOTICKER":
        No valid company in the query.
        
    If the content parts of context has relevent data:
        Overall Sentiment: Positive
        Overall Justification: The Historical documents consistently present Alphabet Inc. (GOOGL) in a positive light. One highlights it as being highly owned by hedge funds, indicating institutional confidence, while the other describes it as a strong momentum stock, pointing to positive market performance expectations.
        Stock Advice: Based on the positive sentiments expressed in the Historical documents, it is advisable to consider purchasing Alphabet Inc. (GOOGL) stock. The positive perceptions among hedge funds and the stock’s status as a strong momentum stock suggest potential for future success.
   
    Else:
        Respond "Company {Company name}Alphabet Inc.(Google) {nasdaq company ticker}(GOOGL) details not found in the Historical Data".
    
    Please follow the steps to analyze the sentiment of each document’s content; and exact structure illustrated in above example response to provide an overall sentiment, justification and give stock purchase advice. Provide only Overall response, don't provide documentwise response or any note. Decorate the reponse with html/css tags.
    
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
    sentiment=extract_between(prediction,"Overall Sentiment: ","Overall Justification:").strip()
    if sentiment == "Positive":
        st.success("Positive : Go Ahead...!")
    elif sentiment == "Negative":
        st.warning("Negative : Don't...!")
    elif sentiment == "Neutral":
        st.info("Neutral : Need to Analyse further")
    st.write( prediction, unsafe_allow_html=True)
    return(cmp_tr)
    
    
def real_time(cmp_tr):
    
    st.info("IN DEVELOPMENT")
    print(cmp_tr)
    # Initialise ChromaDB Database
    chroma_db = DBStorage()
    chroma_db.load_vectors()
    context_for_query = chroma_db.get_context_for_query(cmp_tr,k=5)
    
    
    qna_system_message = """
    You are an assistant to a financial services firm who answers user queries on Stock Investments.
    User input will have the context required by you to answer user questions.
    This context will begin with the token: ###Context.
    The context contains references to specific portions of a document relevant to the user query.

    User questions will begin with the token: ###Question.
    
    First, find the 'nasdaq company ticker' of the related company in the question provided.
    Your task is to perform sentiment analysis on the content part of each documents provided in the Context, which discuss a company identified by its 'nasdaq company ticker'. The goal is to determine the overall sentiment expressed across all documents and provide an overall justification. Based on the sentiment analysis, give a recommendation on whether the company’s stock should be purchased.

    Step-by-Step Instructions:
        1. Read the Context: Carefully read the content parts of each document provided in the list of Documents.
        2. Determine Overall Sentiment: Analyze the sentiment across all documents and categorize the overall sentiment as Positive, Negative, or Neutral.
        3. Provide Overall Justification: Summarize the key points from all documents to justify the overall sentiment.
        4. Stock Advice: Based on the overall sentiment and justification, provide a recommendation on whether the company’s stock should be purchased.

    Example Analysis:
        Context: 
            [Document(metadata={'platform': 'Google News', 'company': 'GOOGL', 'ingestion_timestamp': '2024-10-22T10:11:17.257275', 'word_count': 91}, page_content="{'title': 'Alphabet Inc. (GOOGL): Among the Most Owned Stocks by Hedge Funds Right Now - Insider Monkey', 'content': 'Alphabet Inc. (GOOGL): Among the Most Owned Stocks by Hedge Funds Right Now  Insider Monkey'}"),Document(metadata={'platform': 'Google News', 'company': 'GOOGL', 'ingestion_timestamp': '2024-10-22T10:11:17.257275', 'word_count': 59}, page_content="{'title': 'Here\'s Why Alphabet (GOOGL) is a Strong Momentum Stock - MSN', 'content': 'Here\'s Why Alphabet (GOOGL) is a Strong Momentum Stock  MSN'}")]
    
    Response Format:
    If the content parts of context has relevent data:
        Overall Sentiment: Positive
        Overall Justification: The RealTime documents consistently present Alphabet Inc. (GOOGL) in a positive light. One highlights it as being highly owned by hedge funds, indicating institutional confidence, while the other describes it as a strong momentum stock, pointing to positive market performance expectations.
        Stock Advice: Based on the positive sentiments expressed in the RealTime documents, it is advisable to consider purchasing Alphabet Inc. (GOOGL) stock. The positive perceptions among hedge funds and the stock’s status as a strong momentum stock suggest potential for future success.
   
    Else:
        Respond "Company {Company name}Alphabet Inc.(Google) {nasdaq company ticker}(GOOGL) details not found in the RealTime Data".
    
    Please follow the steps to analyze the sentiment of each document’s content; and exact structure illustrated in above example response to provide an overall sentiment, justification and give stock purchase advice. Provide only Overall response, don't provide documentwise response. Decorate the reponse with html/css tags.
    
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
    sentiment=extract_between(prediction,"Overall Sentiment: ","Overall Justification:").strip()
    # if sentiment == "Positive":
    #     st.success("Positive : Go Ahead...!")
    # elif sentiment == "Negative":
    #     st.warning("Negative : Don't...!")
    # elif sentiment == "Neutral":
    #     st.info("Neutral : Need to Analyse further")
    # st.write( prediction, unsafe_allow_html=True)
    
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
        st.markdown("<h2 class='column-header'>From Real-Time Agent Replies</h2>", unsafe_allow_html=True)
        
        with st.container():
            if user_question:
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
