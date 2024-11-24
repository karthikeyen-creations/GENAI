from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-405B-Instruct", 
	messages=messages, 
	max_tokens=500,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")