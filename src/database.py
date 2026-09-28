import os
import psycopg2

def hybrid_search(query_text: str, query_embedding: list, top_k: int = 5) -> list:
    """Executes Reciprocal Rank Fusion (RRF) directly in PostgreSQL."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "crag_warehouse"),
        user=os.getenv("DB_USER", "crag_admin"),
        password=os.getenv("DB_PASSWORD", "crag_password")
    )
    cur = conn.cursor()
    
    # RRF combines pgvector HNSW dense rank with tsvector GIN sparse rank
    rrf_query = """
    WITH semantic_search AS (
        SELECT id, source_id, document_text,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
        FROM regulatory_docs
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, source_id, document_text,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts_vector, plainto_tsquery('simple', %s)) DESC) AS rank
        FROM regulatory_docs
        WHERE fts_vector @@ plainto_tsquery('simple', %s)
        LIMIT 20
    )
    SELECT
        COALESCE(s.source_id, k.source_id) AS doc_id,
        COALESCE(s.document_text, k.document_text) AS text,
        COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """
    
    cur.execute(rrf_query, (query_embedding, query_text, query_text, top_k))
    results = [{"doc_id": row[0], "text": row[1], "score": float(row[2])} for row in cur.fetchall()]
    
    cur.close()
    conn.close()
    return results
