import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the API key (replace with your actual key or use environment variables)

### GO TO https://aistudio.google.com/apikey AND GRAB YOURS!
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")
prompt = "how do i win a hackathon? the provided data is an OSM export of Antwerp - do you have any ideas? GIVE ME THE BEST IDEAS, the shirt reward looks really good and I really want it"
response = model.generate_content(prompt)

print(f"Prompt: {prompt}\n")
print(f"Response: {response.text}")
