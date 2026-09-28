from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Citation(BaseModel):
    source_id: str = Field(description="The exact DOC_ID of the regulatory circular used.")
    verbatim_quote: str = Field(description="The exact text span copied word-for-word from the context.")
    rationale: str = Field(description="How this specific clause answers the user query.")

class FinalGeneration(BaseModel):
    answer: str = Field(description="The comprehensive regulatory answer.")
    citations: List[Citation] = Field(description="Strict mapping of claims to retrieved documents.")

def generate_node(state):
    """Synthesizes the final answer using retrieved context and enforces citation constraints."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(FinalGeneration)
    
    context_str = "\n\n".join([f"[DOC_ID: {d['doc_id']}]\n{d['text']}" for d in state["documents"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Central Bank compliance AI. Answer the query based strictly on the provided context. You must cite exact DOC_IDs and verbatim quotes for every claim. Do not hallucinate."),
        ("human", "Context:\n{context}\n\nQuery: {question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": state["question"]})
    
    return {"generation": response.answer, "citations": response.citations}
