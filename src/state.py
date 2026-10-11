from typing import List, Dict, TypedDict, Any

class GraphState(TypedDict):
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
