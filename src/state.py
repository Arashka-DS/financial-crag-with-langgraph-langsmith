from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field

class Citation(BaseModel):
    source_doc_id: int = Field(description="The ID of the document used for this specific claim.")
    exact_quote: str = Field(description="The exact verbatim quote from the text that supports the claim.")

class GroundedAnswer(BaseModel):
    answer_text: str = Field(description="The comprehensive financial answer.")
    citations: List[Citation] = Field(description="List of citations mapping claims to source documents.")

# Update the Global State to hold citations
class CRAGState(TypedDict):
    question: str
    transformed_query: str
    documents: List[Dict]
    generation: str
    citations: List[dict] # NEW
    doc_relevance_passed: bool
    hallucination_check_passed: bool
    retry_count: int
    execution_trace: List[str]

# Structured Grading Schemas
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved document chunks."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
    reasoning: str = Field(description="Brief explanation of why the document is relevant or not")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on model generation against facts."""
    binary_score: str = Field(description="Answer is grounded in the provided facts, 'yes' or 'no'")
    explanation: str = Field(description="Verification details confirming groundedness or citing hallucinations")

class GradeAnswer(BaseModel):
    """Binary score to determine if the generation answers the actual user query."""
    binary_score: str = Field(description="Answer addresses the user's question, 'yes' or 'no'")
    critique: str = Field(description="Critique of completeness or missing financial context")
