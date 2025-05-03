import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure the API key (replace with your actual key or use environment variables)
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY environment variable not set.")
    exit()

# Create the client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

prompt = "What food should I buy for a hackathon? I need to make sure that everyone's brains are in top form. Last time everyone seemed to like bananas, pizzas and waffles"
model_name = "tngtech/deepseek-r1t-chimera:free"

# Make the API call
completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": prompt,
        },
    ],
)

# Print the response
print(f"Prompt: {prompt}\n")
print(f"Response: {completion.choices[0].message.content}")
