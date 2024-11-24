import gradio as gr
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

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
            
            Return only the generated PostgreSQL Function.
            """
        user_message = f"###DB2 Stored Procedure\n{db2_code}"
        return self.mistral_chat(system_message, user_message)

    def explain_db2_procedure(self, db2_code):
        system_message = """
            You are an expert in database programming and are tasked with analysing a DB2 Stored Procedure. 
            The input will be the complete code of a DB2 Stored Procedure.
            Your task is to:
            1. Analyze the logic and operations performed by the DB2 Stored Procedure.
            2. Provide a detailed explanation of the DB2 Stored Procedure, highlighting the key steps and functionality.
            """
            
        user_message = f"###DB2 Stored Procedure\n{db2_code}"
        return self.mistral_chat(system_message, user_message)

    def compare_procedures(self, db2_code, postgres_code):
        system_message = """
            You are an expert in database programming and are tasked with comparing a DB2 Stored Procedure and a PostgreSQL function. 
            The input will be the complete codes of a DB2 Stored Procedure and a PostgreSQL function.
            Your task is to:
            1. Analyze the results, logic and operations performed by the DB2 Stored Procedure and the PostgreSQL function.
            2. Compare the DB2 Stored Procedure and the PostgreSQL function, noting any differences in syntax, results, functions, and overall logic.
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

def main():
    # Load Hugging Face API key from environment variable
    api_key = os.getenv("HUGGINGFACE_HUB_TOKEN_SQL_TOOL")
    if api_key is None:
        raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN_SQL_TOOL environment variable with your Hugging Face API token.")
    
    app = DB2ToPostgresConverterApp(api_key=api_key)
    
    # Set up Gradio Interface
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            <h1 style="text-align: center; color: #4CAF50">DB2 to PostgreSQL Converter</h1>
            <p style="text-align: center; color: #888888">
                Convert DB2 Stored Procedures to PostgreSQL Functions and Analyze Their Differences
            </p>
            <hr>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                db2_code_input = gr.Textbox(
                    label="DB2 Stored Procedure", 
                    placeholder="Paste your DB2 Stored Procedure code here...", 
                    lines=10
                )
                convert_button = gr.Button("Convert and Analyze", elem_id="convert-button")
            
            with gr.Column(scale=2):
                postgres_output = gr.Textbox(label="PostgreSQL Function", lines=10, interactive=False)
                explanation_output = gr.Textbox(label="Explanation of DB2 Stored Procedure", lines=10, interactive=False)
                comparison_output = gr.Textbox(label="Comparison Between DB2 and PostgreSQL", lines=10, interactive=False)
        
        # Assign button action
        convert_button.click(
            fn=app.process_db2_to_postgres,
            inputs=db2_code_input,
            outputs=[postgres_output, explanation_output, comparison_output]
        )
    
    # Launch Gradio Interface with custom styling
    interface.launch()

if __name__ == "__main__":
    main()
