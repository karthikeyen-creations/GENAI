import gradio as gr
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os

# Set your Hugging Face token
token = "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"
os.environ["HUGGINGFACE_HUB_TOKEN"] = token

# Initialize the LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.1-405B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def convert_db2_to_postgresql(db2_procedure):
    # Use the model to convert the DB2 Stored Procedure to a PostgreSQL function
    input_prompt = f"Convert the following DB2 Stored Procedure to a PostgreSQL function: {db2_procedure}"
    inputs = tokenizer(input_prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=512)
    postgresql_function = tokenizer.decode(output[0], skip_special_tokens=True)

    # Explain the DB2 Stored Procedure
    explanation_prompt = f"Explain the logic and operations performed by the following DB2 Stored Procedure: {db2_procedure}"
    inputs = tokenizer(explanation_prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=512)
    db2_explanation = tokenizer.decode(output[0], skip_special_tokens=True)

    # Compare the DB2 Stored Procedure and the PostgreSQL function
    comparison_prompt = f"Compare the logic of the following DB2 Stored Procedure and PostgreSQL function. Determine whether both scripts produce the same results when executed under identical conditions. Highlight any differences observed, including syntax, functions, and overall logic.\n\nDB2 Stored Procedure:\n{db2_procedure}\n\nPostgreSQL Function:\n{postgresql_function}"
    inputs = tokenizer(comparison_prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=512)
    comparison_result = tokenizer.decode(output[0], skip_special_tokens=True)

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




x = """
CREATE OR REPLACE PROCEDURE SIMPLE_BONUS_CALC (
    IN P_EMPLOYEE_ID INT,
    OUT P_BONUS DECIMAL(10, 2),
    OUT P_ERROR_CODE INT,
    OUT P_ERROR_MESSAGE VARCHAR(255)
)
LANGUAGE SQL
BEGIN
    DECLARE V_SALARY DECIMAL(10, 2);
    DECLARE V_PERFORMANCE_SCORE DECIMAL(5, 2);
    DECLARE V_BONUS_RATE DECIMAL(5, 2) DEFAULT 0.0;

    -- Default OUT parameters
    SET P_ERROR_CODE = 0;
    SET P_ERROR_MESSAGE = 'Success';

    -- Error handler
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        SET P_ERROR_CODE = SQLCODE;
        SET P_ERROR_MESSAGE = 'Error calculating bonus';
        SET P_BONUS = 0.0;
    END;

    -- Fetch salary and performance score
    SELECT SALARY, PERFORMANCE_SCORE 
    INTO V_SALARY, V_PERFORMANCE_SCORE
    FROM EMPLOYEE
    WHERE EMPLOYEE_ID = P_EMPLOYEE_ID;

    -- Determine bonus rate based on performance score
    IF V_PERFORMANCE_SCORE >= 90 THEN
        SET V_BONUS_RATE = 0.15;
    ELSEIF V_PERFORMANCE_SCORE >= 75 THEN
        SET V_BONUS_RATE = 0.10;
    ELSE
        SET V_BONUS_RATE = 0.05;
    END IF;

    -- Calculate bonus
    SET P_BONUS = V_SALARY * V_BONUS_RATE;
END;

"""