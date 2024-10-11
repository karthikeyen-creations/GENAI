# Import os to handle environment variables
#
# To run this app
# 
# streamlit run <path to this file, for example, rag\streamlit_app_basic.py>
#
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback

load_dotenv()

azure_config = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
    "AZURE_OPENAI_MODEL_NAME": os.getenv("AZURE_OPENAI_MODEL_NAME"),
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION")
    }

st.title("Karthikeyen Assignment4 : Streamlit chat")

with st.sidebar:
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_config["AZURE_OPENAI_ENDPOINT"]
    #openai_api_key = st.text_input("OpenAI API Key", type="password") 
    #os.environ["AZURE_OPENAI_API_KEY"] = azure_config["api-key"]
    #"[Get an Azure OpenAI API key from 'Keys and Endpoint' in Azure Portal](https://portal.azure.com/#blade/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/OpenAI)"

def generate_response(input_text):

    llm = AzureChatOpenAI(temperature=0,
                      api_key=azure_config["AZURE_OPENAI_API_KEY"],
                      openai_api_version=azure_config["AZURE_OPENAI_API_VERSION"],
                      azure_endpoint=azure_config["AZURE_OPENAI_ENDPOINT"],
                      model=azure_config["AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"],
                      validate_base_url=False)

    message = HumanMessage(
        content=input_text
    )
    
    with get_openai_callback() as cb:
        st.info(llm([message]).content) # chat model output
        st.info(cb) # callback output (like cost)

with st.form("my_form"):
    text = st.text_area("Enter text:", "What are the accomplishments of Ratan Tata?")
    submitted = st.form_submit_button("Submit")
    generate_response(text)