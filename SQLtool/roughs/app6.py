import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

class DB2ToPostgresConverterApp:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def mistral_generate(self, prompt, max_length=200):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate text with Mistral
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=max_length, 
                do_sample=True,  
                top_k=50,        
                top_p=0.95,      
                temperature=0.7  
            )
        
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def convert_db2_to_postgres(self, db2_code):
        prompt = f"Convert the following DB2 Stored Procedure to PostgreSQL:\n{db2_code}"
        postgres_code = self.mistral_generate(prompt)
        return postgres_code

    def explain_db2_procedure(self, db2_code):
        prompt = f"Explain the following DB2 Stored Procedure in detail:\n{db2_code}"
        explanation = self.mistral_generate(prompt)
        return explanation

    def compare_procedures(self, db2_code, postgres_code):
        prompt = (
            f"Compare the following DB2 Stored Procedure and PostgreSQL function."
            f"\nDB2 Stored Procedure:\n{db2_code}\n\nPostgreSQL Function:\n{postgres_code}"
            f"\n\nProvide a detailed comparison, noting any differences in syntax, functions, or logic."
        )
        comparison = self.mistral_generate(prompt)
        return comparison

    def process_db2_to_postgres(self, db2_code):
        postgres_code = self.convert_db2_to_postgres(db2_code)
        explanation = self.explain_db2_procedure(db2_code)
        comparison = self.compare_procedures(db2_code, postgres_code)
        return postgres_code, explanation, comparison

def main():
    
    load_dotenv()
    # Load Mistral model and tokenizer with access token from environment
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Replace with the actual model name
    access_token = os.getenv("HUGGINGFACE_HUB_TOKEN_SQL_TOOL")
    
    if access_token is None:
        raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN_SQL_TOOL environment variable with your Hugging Face API token.")

    # Load model and tokenizer with token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=access_token)

    app = DB2ToPostgresConverterApp(model, tokenizer)
    
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
