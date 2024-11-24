import gradio as gr
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI

class DB2ToPostgresConverter:
    def __init__(self, config):
        """
        Initialize the converter with the given configuration and set up models.
        """

        self.config = config
        self.create_models()
        self.client = AzureOpenAI(
            azure_endpoint=self.config["base_url"],
            api_key=self.config["api-key"],
            api_version="2024-02-01"
        )

    def create_models(self):
        """
        Creates and assigns LLM and embedding models to the configuration.
        """
        print("Creating models...")
        llm = AzureChatOpenAI(
            temperature=0,
            api_key=self.config["api-key"],
            openai_api_version=self.config["api_version"],
            azure_endpoint=self.config["base_url"],
            model=self.config["model_deployment"],
            validate_base_url=False
        )
        embedding_model = AzureOpenAIEmbeddings(
            api_key=self.config["api-key"],
            openai_api_version=self.config["api_version"],
            azure_endpoint=self.config["embedding_base_url"],
            model=self.config["embedding_deployment"]
        )
        self.config["models"] = {}
        self.config["models"]["llm"] = llm
        self.config["models"]["embedding_model"] = embedding_model

    def call_model(self, prompt):
        """
        Calls the chat completion model with a given prompt and returns the response.
        """
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response["choices"][0]["message"]["content"]

    def convert_and_explain(self, db2_procedure):
        """
        Convert the DB2 Stored Procedure to a PostgreSQL Function,
        explain the DB2 procedure, and compare the two.
        """
        # Convert DB2 Stored Procedure to PostgreSQL Function
        conversion_prompt = f"Convert the following DB2 stored procedure to a PostgreSQL function:\n{db2_procedure}"
        postgres_function = self.call_model(conversion_prompt)
        
        # Explain the DB2 Stored Procedure
        explanation_prompt = f"Explain the following DB2 stored procedure in detail:\n{db2_procedure}"
        explanation = self.call_model(explanation_prompt)
        
        # Compare DB2 Procedure and PostgreSQL Function
        comparison_prompt = f"Compare the following DB2 stored procedure and PostgreSQL function. Check if they produce the same results and point out any differences:\nDB2 Procedure: {db2_procedure}\nPostgreSQL Function: {postgres_function}"
        comparison = self.call_model(comparison_prompt)
        
        return postgres_function, explanation, comparison


class DB2ToPostgresApp:
    def __init__(self):
        """
        Initializes the Gradio interface and DB2ToPostgresConverter.
        """
        self.config = {
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "embedding_base_url": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                "model_deployment": os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
                "model_name": os.getenv("AZURE_OPENAI_MODEL_NAME"),
                "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
            }
        self.converter = DB2ToPostgresConverter(self.config)

    def gradio_interface(self):
        """
        Defines and launches the Gradio interface for the application.
        """
        with gr.Blocks() as demo:
            # Input and output components for Gradio UI
            db2_input = gr.TextArea(label="DB2 Stored Procedure", placeholder="Enter your DB2 stored procedure here")
            output_postgres = gr.TextArea(label="PostgreSQL Function", interactive=False)
            output_explanation = gr.TextArea(label="DB2 Stored Procedure Explanation", interactive=False)
            output_comparison = gr.TextArea(label="Comparison", interactive=False)
            
            # Add a submit button to trigger the conversion and explanation
            submit_button = gr.Button("Submit")
            submit_button.click(
                self.convert_and_explain, 
                inputs=db2_input, 
                outputs=[output_postgres, output_explanation, output_comparison]
            )
        
        # Launch the Gradio interface
        demo.launch()

    def convert_and_explain(self, db2_procedure):
        """
        Wrapper method for interacting with the DB2ToPostgresConverter.
        """
        # Get the results from the converter
        postgres_function, explanation, comparison = self.converter.convert_and_explain(db2_procedure)
        
        return postgres_function, explanation, comparison


def main():
    """
    The main entry point of the application that starts the Gradio interface.
    """
    
    app = DB2ToPostgresApp()
    app.gradio_interface()


if __name__ == "__main__":
    main()
