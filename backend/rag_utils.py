
from langchain_openai import ChatOpenAI
import os

# Hybrid search: alternate between keyword hits (BM25) and meaning hits (FAISS)
def retrieve(query: str, k: int = 3, vectorstore=None, bm25=None) -> list[str]:
    pool = k * 3                                   # fetch extra, then merge down to k
    bm25.k = pool
    keyword_hits = bm25.invoke(query)              # keyword search
    meaning_hits = vectorstore.similarity_search(query, k=pool)  # meaning search

    best, seen = [], set()
    for pair in zip(keyword_hits, meaning_hits):   # take turns: keyword, meaning
        for d in pair:
            if d.page_content not in seen and len(best) < k:
                seen.add(d.page_content)
                best.append(d)

    return [f"Content:\n{d.page_content}\n\nMetadata:\n{d.metadata}" for d in best]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an assistant for AuroraStay Hotels staff. Staff use you to answer guest questions
    about bookings, cancellations, refunds, charges, rooms, amenities, pets, accessibility and guest conduct.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
        1. Use only information explicitly supported by the retrieved policy information.
        2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
        3. Include all relevant information needed to answer the query completely.
        4. Do not omit important conditions, exceptions, limitations, approval steps, or fees.
        5. Focus only on information directly relevant to the user's query.
        6. Do not include unrelated information. Ignore retrieved chunks that do not relate to the question.
        7. Answer the specific question directly in the first sentence, then give the conditions that apply.
        8. Keep the answer concise, clear, and in plain language for front-desk staff.
        9. Do not invent fees, charges, times, limits, exceptions, or conditions.
        10. If the retrieved information answers only part of the question, answer that part
            and clearly say which part cannot be confirmed.
            Use this sentence ONLY when none of the retrieved information addresses the question:
            "The retrieved policy information does not establish an answer to this question. Check the live property profile or contact the property team."
            If the retrieved information conflicts, clearly mention the conflict.
        11. Property-specific details (fees, parking, breakfast, pets, check-in times, amenities such as pools)
            apply ONLY to the property named. Only say a property has a fee, rule or amenity if the retrieved
            information names that property explicitly. General group rules (e.g. "available only where listed
            in the property profile") do NOT confirm a specific property has it; say it cannot be confirmed.
            Never use a figure from one property for another.
        12. Never promise a refund, waiver, or exception. State who must approve it.
        13. If the retrieved information says to verify the live property profile or reservation confirmation, include that reminder.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 3,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 256,
    vectorstore=None,
    bm25=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore, bm25=bm25)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
