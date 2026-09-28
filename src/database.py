import os
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict

class HybridSearchEngine:
    def __init__(self):
        self.conn_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "financial_knowledge_db"),
            "user": os.getenv("DB_USER", "crag_admin"),
            "password": os.getenv("DB_PASSWORD", "crag_password"),
            "port": 5432
        }

    def get_connection(self):
        conn = psycopg2.connect(**self.conn_params)
        register_vector(conn)
        return conn

    def hybrid_search(self, query_text: str, query_embedding: List[float], top_k: int = 4, rrf_k: int = 60) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. Dense Semantic Search via Cosine Distance
        cursor.execute("""
            SELECT doc_id, title, category, content, 
                   1 - (embedding <=> %s) AS dense_score
            FROM financial_documents
            ORDER BY embedding <=> %s ASC
            LIMIT 10;
        """, (query_embedding, query_embedding))
        dense_results = cursor.fetchall()

        # 2. Sparse Lexical Search via TSVector Full-Text
        cursor.execute("""
            SELECT doc_id, title, category, content, 
                   ts_rank_cd(tsv_content, plainto_tsquery('english', %s)) AS sparse_score
            FROM financial_documents
            WHERE tsv_content @@ plainto_tsquery('english', %s)
            ORDER BY sparse_score DESC
            LIMIT 10;
        """, (query_text, query_text))
        sparse_results = cursor.fetchall()

        cursor.close()
        conn.close()

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_store = {}

        for rank, row in enumerate(dense_results):
            doc_id, title, category, content, _ = row
            doc_store[doc_id] = {"doc_id": doc_id, "title": title, "category": category, "content": content}
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        for rank, row in enumerate(sparse_results):
            doc_id, title, category, content, _ = row
            doc_store[doc_id] = {"doc_id": doc_id, "title": title, "category": category, "content": content}
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{**doc_store[doc_id], "rrf_score": score} for doc_id, score in sorted_docs]
