
# Import necessary libraries
import re
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Import RAG utility functions
from rag_utils import retrieve, rag


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Build the keyword (BM25) search from the same chunks
# ---------------------------------------------------------

STOPWORDS = {"a", "an", "the", "is", "are", "at", "to", "of", "in", "on", "for", "or", "and",
             "what", "which", "how", "do", "does", "can", "i", "my", "it", "be", "with", "if", "by"}

def bm25_words(text):
    """Lowercase, keep codes like 'ash-sea' and prices like '$38', drop common words."""
    return [w for w in re.findall(r"[a-z0-9$\-]+", text.lower()) if w not in STOPWORDS]

# The chunks are already saved inside the FAISS folder (index.pkl)
chunks = list(vectorstore.docstore._dict.values())

bm25 = BM25Retriever.from_documents(chunks, preprocess_func=bm25_words)

print(f"BM25 chunks: {len(chunks)}")


# ---------------------------------------------------------
# Error handling
# ---------------------------------------------------------

def read_request():
    """
    Checks the request BEFORE running search or the LLM.
    Returns (data, None) if the request is valid,
    or (None, error_response) if it is not.
    """
    # silent=True gives None instead of crashing when the body is not JSON
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "The request body must be JSON."}), 400)

    # The question must be text and not empty
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, (jsonify({"error": "Please send a non-empty 'query'."}), 400)

    # k must be a whole number in a sensible range
    k = data.get("k", 3)
    if not isinstance(k, int) or not 1 <= k <= 10:
        return None, (jsonify({"error": "'k' must be a whole number between 1 and 10."}), 400)

    data["query"] = query.strip()
    return data, None


@rag_api.errorhandler(Exception)
def handle_unexpected_error(e):
    """
    Catches anything that breaks during inference (e.g. the OpenAI service is down
    or times out) and returns a clear message instead of a crash page.
    """
    # Keep Flask's normal behaviour for standard web errors (e.g. 404 wrong URL)
    if isinstance(e, HTTPException):
        return e

    # Log the real error for the developer; show a safe message to staff
    print(f"Inference error: {e}")
    return jsonify({
        "error": "The answer service failed. Please try again, or look up the policy manually."
    }), 500


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3
    }

    Returns the top-k relevant chunks retrieved using
    hybrid search (BM25 keywords + FAISS meaning).
    """

    # Check the request first (returns a 400 message if it is invalid)
    data, error = read_request()
    if error:
        return error

    # Extract parameters
    query = data["query"]
    k = data.get("k", 3)

    # Retrieve relevant chunks
    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore,
        bm25=bm25
    )

    # Return the retrieved chunks
    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256
    }

    Returns both the generated answer and the relevant
    chunks used to generate the answer.
    """

    # Check the request first (returns a 400 message if it is invalid)
    data, error = read_request()
    if error:
        return error

    # Extract parameters
    query = data["query"]
    k = data.get("k", 3)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 256)

    # Run the complete RAG pipeline
    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore,
        bm25=bm25
    )

    # Return answer, chunks, and parameters used
    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256
    }

    Returns only the generated answer.
    """

    # Check the request first (returns a 400 message if it is invalid)
    data, error = read_request()
    if error:
        return error

    # Extract parameters
    query = data["query"]
    k = data.get("k", 3)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 256)

    # Run the complete RAG pipeline
    generated_answer, _ = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore,
        bm25=bm25
    )

    # Return only the answer
    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=True)
