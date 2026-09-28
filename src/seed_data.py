import os
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_openai import OpenAIEmbeddings

def bootstrap_database():
    print("Bootstrapping vector database with CBI regulatory circulars...")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "crag_postgres"),
        database=os.getenv("DB_NAME", "crag_warehouse"),
        user=os.getenv("DB_USER", "crag_admin"),
        password=os.getenv("DB_PASSWORD", "crag_password")
    )
    register_vector(conn)
    cur = conn.cursor()
    
    docs = [
        {"source_id": "CBI_CIRCULAR_1401_55", "text": "Under CBI Circular 1401/55, daily peer-to-peer (P2P) card transfer limits are capped at 100 million Rials (10 million Tomans) per national ID, regardless of the number of debit cards the user possesses."},
        {"source_id": "AML_DIRECTIVE_402B", "text": "To mitigate cryptocurrency money laundering, payment gateways are strictly prohibited from settling funds directly to crypto custodian wallets without Tier-2 KYC verification, including a live selfie and biometric national ID match."},
        {"source_id": "SHAPARAK_SLA_09", "text": "Shaparak settlement cycles (Paya) occur strictly four times per working day: 03:45, 10:45, 13:45, and 17:45. Settlements initiated on Fridays will be held in escrow until the Saturday 03:45 cycle."}
    ]
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Defensive Bootstrapping for CI/CD or testing without an API Key
    if not api_key or not api_key.startswith("sk-"):
        print("WARNING: Valid OPENAI_API_KEY not detected. Generating deterministic mock vectors to keep the container alive...")
        # Generate dummy 1536-dimensional vectors for pgvector schema satisfaction
        embeddings_list = [[0.01] * 1536 for _ in range(len(docs))]
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        embeddings_list = [embeddings.embed_query(d["text"]) for d in docs]
    
    for idx, d in enumerate(docs):
        cur.execute("""
            INSERT INTO regulatory_docs (source_id, document_text, embedding) 
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (source_id) DO NOTHING
        """, (d["source_id"], d["text"], embeddings_list[idx]))
        
    conn.commit()
    cur.close()
    conn.close()
    print("Vector database successfully seeded.")

if __name__ == "__main__":
    bootstrap_database()
