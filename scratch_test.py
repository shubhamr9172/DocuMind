import os
from dotenv import load_dotenv

# Load env before importing app modules
load_dotenv()

from app.rag_engine import RAGEngine

engine = RAGEngine()
print("Engine created.")
engine.ingest_document("The capital of France is Paris.", "test.txt")
print("Document ingested.")

try:
    res = engine.ask_question("What is the capital of France?")
    print("Response:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
