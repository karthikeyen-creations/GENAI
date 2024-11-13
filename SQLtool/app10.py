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
            
            Return only the PostgreSQL result alone.
            """
        user_message = f"###DB2 Stored Procedure\n{db2_code}"
        return self.mistral_chat(system_message, user_message)

    def explain_db2_procedure(self, db2_code):
        system_message = """
            You are an expert in database programming and are tasked with analyzing a DB2 Stored Procedure. 
            The input will be the complete code of a DB2 Stored Procedure.
            Your task is to:
            1. Analyze the logic and operations performed by the DB2 Stored Procedure.
            2. Provide a detailed explanation of the DB2 Stored Procedure, highlighting the key steps and functionality.
            
            Format the response in HTML with titles, subtitles, and bullet points for key steps. (for a dark themed background)
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
            
            Format the response in HTML, highlighting syntax differences and logical steps. Finally, Emphasize and give a brief if the result would vary between the two codes. Tabulate, if needed. (for a dark themed background)
            """
        user_message = (
            f"###DB2 Stored Procedure\n{db2_code}\n\n###PostgreSQL Function\n{postgres_code}"
        )
        return self.mistral_chat(system_message, user_message)

    def process_db2_to_postgres(self, db2_code):
        postgres_code = self.convert_db2_to_postgres(db2_code)
        explanation = self.explain_db2_procedure(db2_code)
        comparison = self.compare_procedures(db2_code, postgres_code)
        return postgres_code, explanation, comparison

def remove_overlap_and_concatenate(*texts):
    result = texts[0]
    for i in range(1, len(texts)):
        max_overlap = 0
        min_length = min(len(result), len(texts[i]))
        for j in range(1, min_length + 1):
            if result[-j:] == texts[i][:j]:
                max_overlap = j
        result += texts[i][max_overlap:]
    return result

# Initialize session state if necessary
if 'db2_code_input0' not in st.session_state:
    st.session_state['db2_code_input0'] = ""
    
# Initialize session state for OCR extracted code from images
for i in range(10):
    if f"db2_code_from_image_{i}" not in st.session_state:
        st.session_state[f"db2_code_from_image_{i}"] = ""

# Load Hugging Face API key from environment variable
api_key = os.getenv("HUGGINGFACE_HUB_TOKEN_SQL_TOOL")
if api_key is None:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN_SQL_TOOL environment variable with your Hugging Face API token.")

app = DB2ToPostgresConverterApp(api_key=api_key)

# Sidebar for application information
with st.sidebar:
    st.markdown("""
        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);">
            <h2 style="color: #ffffff; text-align: center;">About the Application</h2>
            <p style="font-size: 16px; color: #f0f0f0; line-height: 1.6; text-align: justify;">
                Welcome to the <span style="color: #80b1c1;"><strong>DB2 to PostgreSQL Converter</strong></span> application. 
                This tool is designed to assist database professionals and developers in seamlessly converting DB2 Stored Procedures to equivalent PostgreSQL functions. 
                It handles both code conversion and analysis, offering insights into key differences between DB2 and PostgreSQL, along with a detailed explanation of each procedure.
            </p>
            <p style="font-size: 16px; color: #f0f0f0; line-height: 1.6; text-align: justify;">
                <strong style="color: #ffb84d;">Key Features:</strong>
                <ul style="color: #f0f0f0; font-size: 14px; list-style-type: disc; padding-left: 20px;">
                    <li><strong style="color: #ffcc00;">DB2 to PostgreSQL Conversion:</strong> Effortlessly convert DB2 Stored Procedures into equivalent PostgreSQL functions while preserving logic and functionality.</li>
                    <li><strong style="color: #ffcc00;">In-depth Code Explanation:</strong> Receive a detailed breakdown of the DB2 procedure, highlighting the steps, logic, and specific operations involved.</li>
                    <li><strong style="color: #ffcc00;">Code Comparison:</strong> Compare the DB2 Stored Procedure with the generated PostgreSQL function, emphasizing any differences in syntax and logic.</li>
                    <li><strong style="color: #ffcc00;">Image Upload Support:</strong> Upload up to <strong style="color: #ffcc00;">10 images</strong> containing DB2 Stored Procedure code. The OCR will automatically extract and concatenate the code.</li>
                    <li><strong style="color: #ffcc00;">Real-time Updates:</strong> Any modifications to the OCR extracted text will dynamically update the concatenated code, ensuring the conversion is based on the most recent input.</li>
                    <li><strong style="color: #ffcc00;">Cross-platform Compatibility:</strong> Although optimized for DB2 to PostgreSQL migration, the app is adaptable to support other database systems.</li>
                </ul>
            </p>
            <p style="font-size: 16px; color: #f0f0f0; line-height: 1.6; text-align: justify;">
                <strong style="color: #ffb84d;">How it Works:</strong>
                <ol style="color: #f0f0f0; font-size: 14px; list-style-type: decimal; padding-left: 20px;">
                    <li><strong style="color: #ffcc00;">Input Method:</strong> Paste your DB2 code directly into the text area, or upload up to <strong style="color: #ffcc00;">10 images</strong> containing DB2 code.</li>
                    <li><strong style="color: #ffcc00;">OCR Processing:</strong> The app will automatically use OCR to extract the code from images and concatenate them into a single code block for conversion.</li>
                    <li><strong style="color: #ffcc00;">Conversion:</strong> The DB2 code is then converted to an equivalent PostgreSQL function while maintaining the original logic and flow.</li>
                    <li><strong style="color: #ffcc00;">Explanation and Comparison:</strong> A detailed explanation of the DB2 code and a comparison between the DB2 and PostgreSQL versions is provided to ensure accuracy.</li>
                    <li><strong style="color: #ffcc00;">Final Output:</strong> You will receive the resulting PostgreSQL function, the explanation, and a comparison of the DB2 and PostgreSQL code.</li>
                </ol>
            </p>
            <p style="font-size: 16px; color: #f0f0f0; line-height: 1.6; text-align: justify;">
                This tool is perfect for those who need to migrate legacy DB2 systems to PostgreSQL, analyze database code, or compare how these two database systems handle similar tasks. By offering both code conversion and deep analysis, it provides a streamlined way to ensure your database migrations are accurate and efficient.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="position: fixed; bottom: 25px; background-color: #1f1f1f; padding: 10px; border-radius: 15px; text-align: center;">
        <p style="color: #cccccc; font-size: 14px; text-align: center; margin: 0;">
            Developed by: <a href="https://www.linkedin.com/in/karthikeyen92/" target="_blank" style="color: #4DA8DA; text-decoration: none;">Karthikeyen Packirisamy</a>
        </p>
    </div>""", unsafe_allow_html=True)

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
tab1, tab2 = st.tabs(["Text Input", "Images Upload"])

# Callback function to update concatenated text whenever any OCR text area changes
def update_concatenated_text():
    concatenated_text = remove_overlap_and_concatenate(*[st.session_state[f"db2_code_from_image_{i}"] for i in range(10)])
    st.session_state["db2_code_input0"] = concatenated_text

# Text Input Tab
with tab1:
    db2_code_input = st.text_area("Enter DB2 Stored Procedure", placeholder="Paste your DB2 Stored Procedure code here...", height=200, value=st.session_state.get("db2_code_input0", ""))

    # Convert and Analyze button
    if st.button("Convert and Analyze", key="convert_button_text") and db2_code_input:
        with st.spinner("Processing..."):
            postgres_code, explanation, comparison = app.process_db2_to_postgres(db2_code_input)
        
        # Display results
        st.write("### PostgreSQL Function")
        st.code(postgres_code, language='sql')
        
        st.write("### Explanation of DB2 Stored Procedure")
        st.markdown(explanation, unsafe_allow_html=True)

        st.write("### Comparison Between DB2 and PostgreSQL")
        st.markdown(comparison, unsafe_allow_html=True)

# Image Upload Tab
with tab2:
    for i in range(10):
        uploaded_image = st.file_uploader(f"Upload image {i+1} containing DB2 Stored Procedure code", type=["png", "jpg", "jpeg"], key=f"image_upload_{i}")
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.session_state[f"db2_code_from_image_{i}"] = pytesseract.image_to_string(image)
        
        # Text area for OCR extracted code with callback
        st.text_area(f"OCR Extracted Code from Image {i+1}", st.session_state[f"db2_code_from_image_{i}"], height=200, key=f"ocr_text_area_{i}", on_change=update_concatenated_text)

    # Display the concatenated result
    st.text_area("Concatenated DB2 Code", st.session_state["db2_code_input0"], height=200)

    # Convert and Analyze button
    if st.button("Convert and Analyze", key="convert_button_image") and st.session_state.get("db2_code_input0", ""):
        with st.spinner("Processing..."):
            postgres_code, explanation, comparison = app.process_db2_to_postgres(st.session_state.get("db2_code_input0", ""))
        
        # Display results
        st.write("### PostgreSQL Function")
        st.code(postgres_code, language='sql')
        
        st.write("### Explanation of DB2 Stored Procedure")
        st.markdown(explanation, unsafe_allow_html=True)

        st.write("### Comparison Between DB2 and PostgreSQL")
        st.markdown(comparison, unsafe_allow_html=True)
