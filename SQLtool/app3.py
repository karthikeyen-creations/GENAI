import os
import gradio as gr
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


# Initialize the Azure OpenAI Chat model
class Config:
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

config = Config()
llm = AzureChatOpenAI(
    temperature=0,
    api_key=config.azure_config["api-key"],
    openai_api_version=config.azure_config["api_version"],
    azure_endpoint=config.azure_config["base_url"],
    model=config.azure_config["model_deployment"],
    validate_base_url=False
)


def convert_db2_to_postgresql(db2_procedure):
    # Use the model to convert the DB2 Stored Procedure to a PostgreSQL function
    input_prompt = f"Convert the following DB2 Stored Procedure to a PostgreSQL function: {db2_procedure}"
    response = llm.chat(input_prompt)
    postgresql_function = response['message']['content']['text']

    # Explain the DB2 Stored Procedure
    explanation_prompt = f"Explain the logic and operations performed by the following DB2 Stored Procedure: {db2_procedure}"
    response = llm.chat(explanation_prompt)
    db2_explanation = response['message']['content']['text']

    # Compare the DB2 Stored Procedure and the PostgreSQL function
    comparison_prompt = f"Compare the logic of the following DB2 Stored Procedure and PostgreSQL function. Determine whether both scripts produce the same results when executed under identical conditions. Highlight any differences observed, including syntax, functions, and overall logic.\n\nDB2 Stored Procedure:\n{db2_procedure}\n\nPostgreSQL Function:\n{postgresql_function}"
    response = llm.chat(comparison_prompt)
    comparison_result = response['message']['content']['text']

    return postgresql_function, db2_explanation, comparison_result


# Create the Gradio interface
demo = gr.Interface(
    convert_db2_to_postgresql,
    inputs=gr.Textbox(label="DB2 Stored Procedure"),
    outputs=[
        gr.Code(label="PostgreSQL Function"),
        gr.Textbox(label="DB2 Stored Procedure Explanation"),
        gr.Textbox(label="Comparison Result"),
    ],
    title="DB2 to PostgreSQL Converter",
    description="Convert DB2 Stored Procedures to PostgreSQL functions and compare their logic.",
)


# Launch the Gradio application
demo.launch()