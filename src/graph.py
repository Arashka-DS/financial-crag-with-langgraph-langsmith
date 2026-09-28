from langgraph.graph import StateGraph, END
from src.state import CRAGState
from src.nodes import (
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    check_hallucinations_node
)

def decide_to_generate(state: CRAGState) -> str:
    """Conditional Edge: Determines if documents are relevant or query must be rewritten."""
    if state["doc_relevance_passed"]:
        return "generate"
    
    if state.get("retry_count", 0) >= 2:
        return "generate" # Cap retries to prevent infinite loops; generate fallback
        
    return "transform_query"

def decide_hallucination_action(state: CRAGState) -> str:
    """Conditional Edge: Confirms grounding before returning to the user."""
    if state["hallucination_check_passed"]:
        return END
    
    if state.get("retry_count", 0) >= 2:
        return END # Stop loops and surface best effort with warning
        
    return "generate"

def build_crag_graph():
    workflow = StateGraph(CRAGState)

    # 1. Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_hallucination", check_hallucinations_node)

    # 2. Build Connections
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "check_hallucination")

    workflow.add_conditional_edges(
        "check_hallucination",
        decide_hallucination_action,
        {
            END: END,
            "generate": "generate"
        }
    )

    return workflow.compile()

crag_app = build_crag_graph()
