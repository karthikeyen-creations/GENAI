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

load_dotenv()

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

st.set_page_config(page_title="Document Genie", layout="wide")
st.title("Karthikeyen Assignment5 - Streamlit PDF Query")

st.markdown("""
1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, models):
    embeddings = models.embedding_model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Assignment5/streamlit/faiss_index")


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

def get_conversational_chain(llm):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# def generate_response(user_question, models):
#     print("Question: ", user_question)
#     new_db = FAISS.load_local("faiss_index", models.embedding_model, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     print(response)
#     st.write("Reply: ", response["output_text"])

def user_input(user_question, models):
    print("Question: ", user_question)
    new_db = FAISS.load_local("Assignment5/streamlit/faiss_index", models.embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(models.llm)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.header("Ask a question")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    if user_question:
        models=create_models(azure_config)
        user_input(user_question, models)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                models=create_models(azure_config)
                get_vector_store(text_chunks, models)
                st.success("Done")
    # with st.form("my_form"):
    #     text = st.text_area("Enter text:", "How much is Vijay's experience?")
    #     submitted = st.form_submit_button("Submit")
    #     print(models)
    #     #create_models(azure_config)
    #     generate_response(text, models)
if __name__ == "__main__":
    main()
