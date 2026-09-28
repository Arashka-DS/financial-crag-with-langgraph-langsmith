from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes import retrieve_node, grade_documents_node, transform_query_node, generate_node

def decide_to_generate(state):
    """Evaluates whether to generate or enter query rewriting, bounded by max retries."""
    if state["web_fallback"]:
        # Guardrail: Break out after 2 failed retries to prevent infinite token loops
        if state.get("retry_count", 0) >= 2:
            return "generate"
        return "transform_query"
    return "generate"

def build_crag_graph():
    workflow = StateGraph(GraphState)
    
    # Register Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)
    
    # Build Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional Branching
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

crag_engine = build_crag_graph()
