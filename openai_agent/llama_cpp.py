import asyncio
from llama_cpp import Llama

# Load the model
llm = Llama(model_path="path/to/your/model.bin")

# Async generator for streaming tokens
async def async_generate(prompt):
    for token in await asyncio.to_thread(llm, prompt, stream=True):
        yield token

# Chat function
async def chat():
    print("AI Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        print("AI: ", end='', flush=True)
        async for token in async_generate(user_input):
            print(token, end='', flush=True)
        print()  # Newline after response

# Run the chatbot
asyncio.run(chat())