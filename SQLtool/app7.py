from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

db2_stored_procedure = """

"""

sql_system_message = """
You are an expert in database programming and are tasked with converting a DB2 Stored Procedure into a PostgreSQL function. 
        The input will be the complete code of a DB2 Stored Procedure.
        Your task is to:
        1. Analyze the logic and operations performed by the DB2 Stored Procedure.
        2. Generate an equivalent PostgreSQL function that performs the same operations.
"""
sql_user_message = f"""
###DB2 Stored Procedure
        {db2_stored_procedure}
"""
messages= [
            {'role': 'system', 'content': sql_system_message},
            {'role': 'user', 'content': sql_user_message}
        ]
stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2", 
	messages=messages, 
	max_tokens=4096,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")