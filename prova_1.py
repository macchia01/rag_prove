from langchain_ollama import OllamaLLM

# Initialize Ollama with your chosen model
llm = OllamaLLM(model="llama3.2")

# Invoke the model with a query
response = llm.invoke("What is LLM?")
print(response)
