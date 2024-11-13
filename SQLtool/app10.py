import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# Load environment variables from .env file
load_dotenv()

class DB2ToPostgresConverterApp:
    def __init__(self, api_key):
        self.client = InferenceClient(api_key=api_key)

    def mistral_chat(self, system_message, user_message):
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]
        
        response = ""
        stream = self.client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            max_tokens=4096,
            stream=True
        )
        
        for chunk in stream:
            response += chunk.choices[0].delta.content
        return response

    def convert_db2_to_postgres(self, db2_code):
        system_message = """
            You are an expert in database programming and are tasked with converting a DB2 Stored Procedure into a PostgreSQL function. 
            The input will be the complete code of a DB2 Stored Procedure.
            Your task is to:
            1. Analyze the logic and operations performed by the DB2 Stored Procedure.
            2. Generate an equivalent PostgreSQL function that performs the same operations.
            
            Return the PostgreSQL result alone.
            """
        user_message = f"###DB2 Stored Procedure\n{db2_code}"
        print(user_message)
        return self.mistral_chat(system_message, user_message)

    def explain_db2_procedure(self, db2_code):
        system_message = """
            You are an expert in database programming and are tasked with analyzing a DB2 Stored Procedure. 
            The input will be the complete code of a DB2 Stored Procedure.
            Your task is to:
            1. Analyze the logic and operations performed by the DB2 Stored Procedure.
            2. Provide a detailed explanation of the DB2 Stored Procedure, highlighting the key steps and functionality.
            
            Format the response in HTML with titles, subtitles, and bullet points for key steps.
            """
        user_message = f"###DB2 Stored Procedure\n{db2_code}"
        return self.mistral_chat(system_message, user_message)

    def compare_procedures(self, db2_code, postgres_code):
        system_message = """
            You are an expert in database programming and are tasked with comparing a DB2 Stored Procedure and a PostgreSQL function. 
            Both databases have the same Configuration settings and data. The input will be the complete codes of a DB2 Stored Procedure and a PostgreSQL function.
            Your task is to:
            1. Analyze the results, logic, and operations performed by the DB2 Stored Procedure and the PostgreSQL function.
            2. Compare the DB2 Stored Procedure and the PostgreSQL function, noting any differences in syntax, results, functions, and overall logic.
            
            Format the response in HTML, highlighting syntax differences and logical steps. Emphasize if the result would vary.
            """
        user_message = (
            f"###DB2 Stored Procedure\n{db2_code}\n\n###PostgreSQL Function\n{postgres_code}"
        )
        print(user_message)
        return self.mistral_chat(system_message, user_message)

    def process_db2_to_postgres(self, db2_code):
        postgres_code = self.convert_db2_to_postgres(db2_code)
        explanation = self.explain_db2_procedure(db2_code)
        comparison = self.compare_procedures(db2_code, postgres_code)
        return postgres_code, explanation, comparison

# Load Hugging Face API key from environment variable
api_key = os.getenv("HUGGINGFACE_HUB_TOKEN_SQL_TOOL")
if api_key is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN_SQL_TOOL environment variable with your Hugging Face API token.")

app = DB2ToPostgresConverterApp(api_key=api_key)

# Sidebar for application information
with st.sidebar:
    st.markdown("""
        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);">
            <h2 style="color: #e6e6e6; text-align: center;">About the Application</h2>
            <p style="font-size: 16px; color: #d9d9d9; line-height: 1.6; text-align: justify;">
                This application converts <span style="color: #80b1c1;"><strong>DB2 Stored Procedures</strong></span> to equivalent 
                <span style="color: #d3b673;"><strong>PostgreSQL Functions</strong></span> while offering in-depth explanations 
                and comparisons. It’s an ideal tool for database professionals and developers transitioning between 
                these platforms.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main App Layout
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50">DB2 to PostgreSQL Converter</h1>
    <p style="text-align: center; color: #888888">
        Convert DB2 Stored Procedures to PostgreSQL Functions and Analyze Their Differences
    </p>
    <hr>
    """, unsafe_allow_html=True
)

# Tabs for input options
tab1, tab2 = st.tabs(["Text Input", "Image Upload"])

# Input Variable for DB2 Code
db2_code_input = ""

with tab1:
    db2_code_input = st.text_area("Enter DB2 Stored Procedure", placeholder="Paste your DB2 Stored Procedure code here...", height=200)

with tab2:
    uploaded_image = st.file_uploader("Upload an image containing DB2 Stored Procedure code", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        # Use OCR to extract text from the uploaded image
        image = Image.open(uploaded_image)
        db2_code_from_image = pytesseract.image_to_string(image)
        
        st.write("### Extracted DB2 Code from Image")
        st.text_area("OCR Extracted Code", db2_code_from_image, height=200)  # Display extracted text in a text area
        
        db2_code_input = db2_code_from_image  # Set the extracted text as input

# Process conversion when button is clicked
if st.button("Convert and Analyze") and db2_code_input:
    with st.spinner("Processing..."):
        postgres_code, explanation, comparison = app.process_db2_to_postgres(db2_code_input)
    
    # Display results
    st.write("### PostgreSQL Function")
    st.code(postgres_code, language='sql')

    st.write("### Explanation of DB2 Stored Procedure")
    st.markdown(explanation, unsafe_allow_html=True)

    st.write("### Comparison Between DB2 and PostgreSQL")
    st.markdown(comparison, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("""
    <div style="position: fixed; bottom: 25px; background-color: #1f1f1f; padding: 10px; border-radius: 15px; text-align: center;">
        <p style="color: #cccccc; font-size: 14px; text-align: center; margin: 0;">
            Developed by: <a href="https://www.linkedin.com/in/karthikeyen92/" target="_blank" style="color: #4DA8DA; text-decoration: none;">Karthikeyen Packirisamy</a>
        </p>
    </div>
""", unsafe_allow_html=True)
