import os
from dotenv import load_dotenv

# Load env before importing app modules
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

print(f"Testing model: gemini-flash-latest with key: {os.getenv('GOOGLE_API_KEY')[:10]}...")

try:
    res = llm.invoke("Hello, say 'Test successful'")
    print("Response:", res.content)
except Exception as e:
    import traceback
    traceback.print_exc()
