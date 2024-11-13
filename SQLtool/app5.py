import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI

load_dotenv()

class DB2toPostgresConverter:
    def __init__(self):
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
        
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_config["base_url"],
            api_key=self.azure_config["api-key"],
            api_version="2024-02-01"
        )

    def convert_to_postgresql(self, db2_stored_procedure):
        """
        Converts a DB2 Stored Procedure into a PostgreSQL function using the AzureChatOpenAI model.
        """
        system_message = """
        You are an expert in database programming and are tasked with converting a DB2 Stored Procedure into a PostgreSQL function. 
        The input will be the complete code of a DB2 Stored Procedure.
        Your task is to:
        1. Analyze the logic and operations performed by the DB2 Stored Procedure.
        2. Generate an equivalent PostgreSQL function that performs the same operations.
        """

        user_message = f"""
        ###DB2 Stored Procedure
        {db2_stored_procedure}
        """

        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=4096
        )

        result = response.choices[0].message.content.strip()
        return result

    def compare_functions(self, db2_stored_procedure, postgresql_function):
        """
        Compares the DB2 Stored Procedure and the generated PostgreSQL function.
        """
        system_message = """
        You are an expert in database programming and have been asked to compare a DB2 Stored Procedure and a PostgreSQL function.
        Your task is to:
        1. Carefully read through the provided DB2 Stored Procedure and PostgreSQL function.
        2. Compare the logic implemented in both scripts, noting any differences in their approach.
        3. Confirm whether both scripts will produce the same outputs when executed under the same conditions.
        4. Provide a detailed explanation, only pointing out any differences that may affect the results, including syntax, functions, and overall logic.
        """

        user_message = f"""
        ###DB2 Stored Procedure
        {db2_stored_procedure}
        
        ###PostgreSQL Function
        {postgresql_function}
        """

        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=4096
        )

        result = response.choices[0].message.content.strip()
        return result

def main():
    converter = DB2toPostgresConverter()

    with gr.Blocks() as demo:
        gr.Markdown("# DB2 to PostgreSQL Converter")

        with gr.Tab("Convert to PostgreSQL"):
            db2_stored_procedure = gr.TextArea(label="DB2 Stored Procedure")
            convert_button = gr.Button("Convert to PostgreSQL")
            postgresql_function = gr.TextArea(label="PostgreSQL Function")

            convert_button.click(converter.convert_to_postgresql, inputs=db2_stored_procedure, outputs=postgresql_function)

        with gr.Tab("Comparison"):
            db2_stored_procedure = gr.TextArea(label="DB2 Stored Procedure")
            postgresql_function = gr.TextArea(label="PostgreSQL Function")
            compare_button = gr.Button("Compare Functions")
            comparison_result = gr.TextArea(label="Comparison Result")

            compare_button.click(converter.compare_functions, inputs=[db2_stored_procedure, postgresql_function], outputs=comparison_result)

    demo.launch()

if __name__ == "__main__":
    main()