from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from src.state import CRAGState, GradeDocuments, GradeHallucinations, GradeAnswer
from src.database import HybridSearchEngine
import os

db_engine = HybridSearchEngine()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def retrieve_node(state: CRAGState) -> Dict:
    """Retrieves documents via hybrid search using the current question."""
    query = state.get("transformed_query") or state["question"]
    query_emb = embeddings_model.embed_query(query)
    docs = db_engine.hybrid_search(query_text=query, query_embedding=query_emb, top_k=3)
    
    trace = state.get("execution_trace", [])
    trace.append(f"Retrieved {len(docs)} chunks via Hybrid pgvector + TSVector Search.")
    return {"documents": docs, "execution_trace": trace}

def grade_documents_node(state: CRAGState) -> Dict:
    """Evaluates whether retrieved documents are relevant to the user query."""
    question = state["question"]
    docs = state["documents"]
    trace = state.get("execution_trace", [])

    structured_grader = llm.with_structured_output(GradeDocuments)
    system_prompt = (
        "You are a regulatory compliance auditor. Grade if the document "
        "contains relevant facts to answer the user question. Return 'yes' or 'no'."
    )
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Question: {question}\n\nDocument:\n{document}")
    ])
    grader_chain = grader_prompt | structured_grader

    filtered_docs = []
    has_relevant = False

    for doc in docs:
        result = grader_chain.invoke({"question": question, "document": doc["content"]})
        if result.binary_score.lower() == "yes":
            filtered_docs.append(doc)
            has_relevant = True
        else:
            trace.append(f"Filtered irrelevant doc ID {doc['doc_id']}: {result.reasoning}")

    trace.append(f"Grading complete: {len(filtered_docs)}/{len(docs)} documents retained.")
    return {
        "documents": filtered_docs,
        "doc_relevance_passed": has_relevant,
        "execution_trace": trace
    }

def transform_query_node(state: CRAGState) -> Dict:
    """Rewrites the query to improve retrieval if previous chunks were graded irrelevant."""
    question = state["question"]
    trace = state.get("execution_trace", [])
    retry_count = state.get("retry_count", 0) + 1

    rewrite_prompt = ChatPromptTemplate.from_template(
        "You are a financial search query optimizer. The user asked: '{question}'.\n"
        "Previous document retrieval failed to find relevant financial context.\n"
        "Formulate a more precise, semantically keyword-rich query targeting regulatory or banking documentation."
    )
    rewriter = rewrite_prompt | llm
    new_query = rewriter.invoke({"question": question}).content.strip()

    trace.append(f"Corrective Loop #{retry_count}: Query rewritten to: '{new_query}'")
    return {
        "transformed_query": new_query,
        "retry_count": retry_count,
        "execution_trace": trace
    }

def generate_node(state: CRAGState) -> Dict:
    """Synthesizes the final grounded financial answer with exact citations."""
    question = state["question"]
    docs = state["documents"]
    trace = state.get("execution_trace", [])

    # Pass document IDs along with content so the LLM can cite them
    context = "\n\n".join([f"[DOC_ID: {d['doc_id']} | Category: {d['category']}] {d['content']}" for d in docs])
    
    gen_prompt = ChatPromptTemplate.from_template(
        "You are an enterprise financial regulatory analyst. Answer the user question strictly based on the provided context.\n"
        "You MUST provide citations linking your claims back to the DOC_ID.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    # Enforce Pydantic Structured Output for the generation itself
    structured_generator = llm.with_structured_output(GroundedAnswer)
    generator_chain = gen_prompt | structured_generator
    
    result = generator_chain.invoke({"context": context, "question": question})

    trace.append(f"Synthesized response containing {len(result.citations)} exact citations.")
    
    return {
        "generation": result.answer_text,
        "citations": [cit.dict() for cit in result.citations],
        "execution_trace": trace
    }
    
def check_hallucinations_node(state: CRAGState) -> Dict:
    """Checks if generation is strictly grounded in the retrieved documents."""
    docs = state["documents"]
    generation = state["generation"]
    trace = state.get("execution_trace", [])

    context = "\n\n".join([d["content"] for d in docs])
    structured_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_prompt = ChatPromptTemplate.from_template(
        "Assess whether the generated answer is strictly grounded in the source facts:\n\n"
        "Facts:\n{context}\n\n"
        "Answer:\n{generation}"
    )
    grader = hallucination_prompt | structured_grader
    result = grader.invoke({"context": context, "generation": generation})

    passed = result.binary_score.lower() == "yes"
    trace.append(f"Hallucination Audit: {'PASSED (Grounded)' if passed else 'FAILED (Hallucinated)'}. Details: {result.explanation}")
    return {"hallucination_check_passed": passed, "execution_trace": trace}
