"""
RAG Engine — The Brain of DocuMind
====================================
This module implements the full RAG (Retrieval-Augmented Generation) pipeline.

╔══════════════════════════════════════════════════════════════╗
║                    HOW RAG WORKS                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  INGEST PHASE:                                               ║
║  Document → Split into Chunks → Embed each Chunk → Store     ║
║                                                              ║
║  QUERY PHASE:                                                ║
║  Question → Embed → Find Similar Chunks → Send to LLM → Answer ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Key Terms:
- Embedding: Converting text into numbers (vectors) that capture meaning
- Vector DB: A database optimized for finding similar vectors
- Chunk: A small piece of a document (e.g., 500 characters)
- Retriever: Finds the most relevant chunks for a given question
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import Config


class RAGEngine:
    """
    The RAG Engine — like a smart librarian who:
    1. Organizes books (documents) on shelves (vector database)
    2. When asked a question, finds the most relevant pages
    3. Reads those pages and gives you a summarized answer
    """

    def __init__(self, config=None):
        """
        Initialize all RAG components.

        This sets up:
        - Embeddings model (converts text → vectors, runs LOCALLY)
        - ChromaDB (stores vectors, runs LOCALLY)
        - Text splitter (breaks documents into chunks)
        - LLM (Google Gemini, needs API key)
        """
        self.config = config or Config

        # ─── EMBEDDINGS MODEL ───────────────────────────────
        # Converts text into vectors (lists of numbers).
        # "king" and "queen" get similar vectors because they're related.
        # Model: all-MiniLM-L6-v2 (~80MB, runs on CPU, FREE)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL
        )

        # ─── VECTOR DATABASE (ChromaDB) ─────────────────────
        # Stores document embeddings and enables similarity search.
        # Think of it as a smart filing cabinet that understands meaning.
        self.vectorstore = Chroma(
            collection_name="documind",
            embedding_function=self.embeddings,
            persist_directory=self.config.CHROMA_PERSIST_DIR,
        )

        # ─── TEXT SPLITTER ───────────────────────────────────
        # Splits documents into small, overlapping chunks.
        # Why overlap? So we don't cut sentences in half!
        #
        # Example with chunk_size=100, overlap=20:
        #   Chunk 1: "Python is a programming language. It was created..."
        #   Chunk 2: "It was created by Guido van Rossum in 1991..."
        #   (Notice "It was created" appears in both — that's the overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # ─── LLM (Large Language Model) ─────────────────────
        # Google Gemini Flash — fast, capable, free tier (15 RPM)
        # temperature=0.3 → More factual, less creative
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.LLM_MODEL,
            google_api_key=self.config.GOOGLE_API_KEY,
            temperature=0.3,
        )

        # ─── PROMPT TEMPLATE ────────────────────────────────
        # This tells the LLM HOW to answer using retrieved context.
        # Without this, the LLM would just use its own knowledge
        # (which might be outdated or wrong for your specific docs).
        self.prompt = ChatPromptTemplate.from_template(
            """You are DocuMind, a helpful AI assistant that answers questions
based ONLY on the provided context from uploaded documents.

RULES:
1. Only use information from the CONTEXT below
2. If the context doesn't contain the answer, say "I don't have enough
   information in the uploaded documents to answer this question."
3. Cite the source document name when possible
4. Be concise but thorough

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        )

    def ingest_document(self, text, filename="untitled.txt"):
        """
        Ingest a document into the RAG pipeline.

        Flow: Document Text → Split into Chunks → Embed → Store in ChromaDB

        Args:
            text: The document content as a string
            filename: Name of the document (used for source tracking)

        Returns:
            dict with ingestion stats (filename, chunks_created, total_characters)
        """
        # Step 1: Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Step 2: Create metadata for each chunk
        # This lets us track which document each chunk came from
        metadatas = [
            {"source": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Step 3: Add to ChromaDB
        # Under the hood, this: embeds each chunk → stores vector + text + metadata
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)

        return {
            "filename": filename,
            "chunks_created": len(chunks),
            "total_characters": len(text),
        }

    def ask_question(self, question, top_k=3):
        """
        Ask a question and get an AI answer grounded in your documents.

        Flow: Question → Embed → Find Similar Chunks → Build Prompt → LLM → Answer

        Args:
            question: The user's question
            top_k: Number of relevant chunks to retrieve (default: 3)

        Returns:
            dict with answer, sources list, and chunks_used count
        """
        # Step 1-2: Find the most similar chunks to the question
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            return {
                "answer": "No documents found. Please ingest some documents first.",
                "sources": [],
                "chunks_used": 0,
            }

        # Step 3: Build context string from retrieved chunks
        context = "\n\n---\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in retrieved_docs
            ]
        )

        # Step 4: Send context + question to LLM
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})

            # Collect unique source document names
            sources = list(
                set(doc.metadata.get("source", "unknown") for doc in retrieved_docs)
            )

            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(retrieved_docs),
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "chunks_used": len(retrieved_docs),
            }

    def list_documents(self):
        """List all documents currently stored in the vector database."""
        collection = self.vectorstore._collection
        result = collection.get()

        if not result["metadatas"]:
            return {"documents": [], "total_chunks": 0}

        documents = list(
            set(meta.get("source", "unknown") for meta in result["metadatas"])
        )

        return {"documents": documents, "total_chunks": len(result["ids"])}

    def clear_documents(self):
        """Clear ALL documents from the vector database."""
        collection = self.vectorstore._collection
        count = collection.count()

        if count > 0:
            all_ids = collection.get()["ids"]
            collection.delete(ids=all_ids)

        return {"cleared": True, "chunks_removed": count}
