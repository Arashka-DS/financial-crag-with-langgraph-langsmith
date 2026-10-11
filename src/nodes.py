import os
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from src.database import hybrid_search

# Pydantic Schemas for Structured Output
class GradeDocuments(BaseModel):
    """Binary score for document relevance."""
    binary_score: str = Field(description="Relevance verdict: 'yes' if document is relevant, 'no' if irrelevant.")

class Citation(BaseModel):
    source_id: str = Field(description="Exact DOC_ID of the regulatory circular.")
    verbatim_quote: str = Field(description="Exact verbatim sentence quoted directly from the text.")
    rationale: str = Field(description="Contextual explanation of how this citation grounds the claim.")

class GroundedAnswer(BaseModel):
    answer: str = Field(description="Synthesized regulatory compliance response.")
    citations: List[Citation] = Field(default_factory=list, description="All supporting citations with verbatim quotes. MUST be a valid JSON list/array.")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Answer is entirely grounded in the facts, 'yes' or 'no'")
    explanation: str = Field(description="Explanation of why the answer is grounded or hallucinated")

def retrieve_node(state):
    """Retrieves relevant clauses using native PostgreSQL Reciprocal Rank Fusion."""
    question = state["question"]
    embeddings = OpenAIEmbeddings(
        model="nomic-embed-text",
        check_embedding_ctx_length=False
    )
    query_vector = embeddings.embed_query(question)
    
    docs = hybrid_search(query_text=question, query_embedding=query_vector, top_k=4)
    return {"documents": docs, "retry_count": state.get("retry_count", 0)}

def grade_documents_node(state):
    """Filters out irrelevant chunks to prevent context pollution."""
    llm = ChatOpenAI(model="llama3.2", temperature=0).with_structured_output(GradeDocuments)
    question = state["question"]
    documents = state.get("documents", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert compliance auditor. Grade whether the document contains information directly relevant to the user query. Output 'yes' or 'no'."),
        ("human", "Document: {document}\n\nQuery: {question}")
    ])
    
    chain = prompt | llm
    valid_docs = []
    
    for doc in documents:
        res = chain.invoke({"document": doc["text"], "question": question})
        
        # Fault-tolerant parsing: Handle both dict (Ollama) and Pydantic (OpenAI) returns
        score = res.get("binary_score", "") if isinstance(res, dict) else getattr(res, "binary_score", "")
        
        if "yes" in str(score).lower():
            valid_docs.append(doc)
            
    fallback_needed = len(valid_docs) == 0
    return {"documents": valid_docs, "web_fallback": fallback_needed}

def transform_query_node(state):
    """Rewrites the query using financial taxonomy and increments the loop counter."""
    llm = ChatOpenAI(model="llama3.2", temperature=0)
    question = state["question"]
    current_retry = state.get("retry_count", 0) + 1
    
    prompt = ChatPromptTemplate.from_messages([
        # Added a strict negative constraint to prevent conversational fluff
        ("system", "You are a financial query optimizer. Rewrite the query to optimize for regulatory search, standardizing banking terminology, circular names, and acronyms. ONLY output the optimized query string without any conversational text or quotes."),
        ("human", "Initial Query: {question}\n\nFormulate an optimized semantic search query:")
    ])
    
    # Properly chain the prompt to preserve System/Human message roles in the API call
    better_query = (prompt | llm).invoke({"question": question}).content
    
    return {"question": better_query, "retry_count": current_retry}

def generate_node(state):
    """Synthesizes the answer while forcing strict Pydantic citation grounding."""
    llm = ChatOpenAI(model="llama3.2", temperature=0).with_structured_output(GroundedAnswer)
    question = state["question"]
    docs = state.get("documents", [])
    
    if not docs:
        return {
            "generation": "No official regulatory circular or central bank directive matches this query. Operational actions cannot be confirmed.",
            "citations": []
        }
        
    context_str = "\n\n".join([f"[{d['doc_id']}]\n{d['text']}" for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Central Bank Compliance Officer. Provide an authoritative answer based ONLY on the provided regulatory clauses. For every statement, supply a verbatim quote and its exact DOC_ID. Never extrapolate. ALWAYS return the 'citations' field as a JSON array/list."),
        ("human", "Clauses:\n{context}\n\nQuery: {question}")
    ])
    
    try:
        res = (prompt | llm).invoke({"context": context_str, "question": question})
    except Exception as e:
        # Graceful degradation if the 3B model hallucinates the JSON structure
        return {
            "generation": "SYSTEM SAFEGUARD: The local model formulated an answer but failed to format the citations into a valid JSON schema. Please retry the query.",
            "citations": []
        }
    
    # Fault-tolerant parsing
    if isinstance(res, dict):
        answer = res.get("answer", "")
        citations = res.get("citations", [])
    else:
        answer = getattr(res, "answer", "")
        citations = [c.dict() for c in getattr(res, "citations", [])]
        
    return {"generation": answer, "citations": citations}

def check_hallucinations_node(state):
    """Audits the generated answer against the source documents."""
    llm = ChatOpenAI(model="llama3.2", temperature=0).with_structured_output(GradeHallucinations)
    docs = state.get("documents", [])
    generation = state.get("generation", "")
    
    if not docs or not generation:
        return {"hallucination_check_passed": False}
        
    context_str = "\n\n".join([f"[{d['doc_id']}]\n{d['text']}" for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        # Softened the prompt so the 3B model doesn't reject everything
        ("system", "You are a grader assessing whether an answer is grounded in the provided documents. Output 'yes' if the answer is factually supported by the documents. Output 'no' ONLY if the answer invents or fabricates facts, numbers, or rules not found in the text. Ignore minor conversational filler."),
        ("human", "Documents:\n{context}\n\nGenerated Answer:\n{generation}")
    ])
    
    try:
        res = (prompt | llm).invoke({"context": context_str, "generation": generation})
    except Exception:
        # If the structured output parser fails here, default to blocking the output to be safe
        return {
            "hallucination_check_passed": False,
            "generation": "SYSTEM SAFEGUARD: Hallucination audit failed to parse. Output blocked.",
            "citations": []
        }
    
    # Fault-tolerant parsing
    score = res.get("binary_score", "") if isinstance(res, dict) else getattr(res, "binary_score", "")
    
    # Require an explicit "yes" to pass, rather than just checking for "no"
    if "yes" not in str(score).lower():
        return {
            "hallucination_check_passed": False,
            "generation": "SYSTEM SAFEGUARD: The generated response failed the hallucination audit and was blocked. Unverified claims detected.",
            "citations": []
        }
        
    return {"hallucination_check_passed": True}
