from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field

class Citation(BaseModel):
    source_doc_id: int = Field(description="The ID of the document chunk supporting the claim.")
    exact_quote: str = Field(description="The exact verbatim quote from the text that supports the claim.")

class GroundedAnswer(BaseModel):
    answer_text: str = Field(description="The comprehensive financial answer.")
    citations: List[Citation] = Field(default_factory=list, description="List of citations mapping claims to source documents.")

class CRAGState(TypedDict):
    question: str
    transformed_query: str
    documents: List[Dict[str, Any]]
    generation: str
    citations: List[Dict[str, Any]]
    doc_relevance_passed: bool
    hallucination_check_passed: bool
    retry_count: int
    execution_trace: List[str]
    web_fallback: bool

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
    reasoning: str = Field(description="Brief explanation of why the document is relevant or not")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the provided facts, 'yes' or 'no'")
    explanation: str = Field(description="Verification details confirming groundedness or citing hallucinations")
