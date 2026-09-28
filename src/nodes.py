import os
from typing import List
from pydantic import BaseModel, Field
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
    citations: List[Citation] = Field(description="All supporting citations with verbatim quotes.")

def retrieve_node(state):
    """Retrieves relevant clauses using native PostgreSQL Reciprocal Rank Fusion."""
    question = state["question"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = embeddings.embed_query(question)
    
    docs = hybrid_search(query_text=question, query_embedding=query_vector, top_k=4)
    return {"documents": docs, "retry_count": state.get("retry_count", 0)}

def grade_documents_node(state):
    """Filters out irrelevant chunks to prevent context pollution."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeDocuments)
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert compliance auditor. Grade whether the document contains information directly relevant to the user query. Output 'yes' or 'no'."),
        ("human", "Document: {document}\n\nQuery: {question}")
    ])
    
    chain = prompt | llm
    valid_docs = []
    
    for doc in documents:
        res = chain.invoke({"document": doc["text"], "question": question})
        if res.binary_score.lower() == "yes":
            valid_docs.append(doc)
            
    # Trigger fallback if no documents passed the grade
    fallback_needed = len(valid_docs) == 0
    return {"documents": valid_docs, "web_fallback": fallback_needed}

def transform_query_node(state):
    """Rewrites the query using financial taxonomy and increments the loop counter."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question = state["question"]
    current_retry = state.get("retry_count", 0) + 1
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial query optimizer. Rewrite the query to optimize for regulatory search, standardizing banking terminology, circular names, and acronyms."),
        ("human", "Initial Query: {question}\n\nFormulate an optimized semantic search query:")
    ])
    
    better_query = llm.invoke(prompt.format(question=question)).content
    return {"question": better_query, "retry_count": current_retry}

def generate_node(state):
    """Synthesizes the answer while forcing strict Pydantic citation grounding."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GroundedAnswer)
    question = state["question"]
    docs = state["documents"]
    
    if not docs:
        return {
            "generation": "No official regulatory circular or central bank directive matches this query. Operational actions cannot be confirmed.",
            "citations": []
        }
        
    context_str = "\n\n".join([f"[{d['doc_id']}]\n{d['text']}" for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Central Bank Compliance Officer. Provide an authoritative answer based ONLY on the provided regulatory clauses. For every statement, supply a verbatim quote and its exact DOC_ID. Never extrapolate."),
        ("human", "Clauses:\n{context}\n\nQuery: {question}")
    ])
    
    res = (prompt | llm).invoke({"context": context_str, "question": question})
    return {"generation": res.answer, "citations": [c.dict() for c in res.citations]}
