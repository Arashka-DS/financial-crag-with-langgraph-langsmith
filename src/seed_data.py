import os
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_openai import OpenAIEmbeddings

def seed_regulatory_corpus():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    docs = [
        {
            "title": "CBI Circular 402/12 - Digital Asset Gateway Ceilings",
            "category": "CBI_REGULATION",
            "content": "Under Central Bank Circular 402/12, all domestic payment settlement gateways servicing digital asset exchanges must enforce a maximum cumulative daily fiat on-ramp limit of 250,000,000 IRR per national ID across all verified cards. Accounts failing Tier-2 biometric KYC are capped at 50,000,000 IRR per 24-hour rolling window."
        },
        {
            "title": "Interbank Settlement Framework (Satna & Paya Cycles)",
            "category": "SETTLEMENT_POLICY",
            "content": "Paya interbank batch settlement cycles execute four times daily on business days: 03:45, 10:45, 13:45, and 17:45. Satna real-time gross settlement operates continuously between 08:00 and 13:00 for individual transfers exceeding 500,000,000 IRR, requiring explicit Sheba (IBAN) destination validation."
        },
        {
            "title": "AML Directive 88 - High-Velocity Merchant Thresholds",
            "category": "AML_COMPLIANCE",
            "content": "Anti-Money Laundering Directive 88 mandates immediate automated suspension of merchant POS and IPG endpoints if transaction velocity exceeds 40 transactions per minute with identical card issuers, or if total debit turnaround exceeds 300% of historical 30-day baseline without prior invoice declaration."
        },
        {
            "title": "Crypto Asset Custody & Cold Storage Minimums",
            "category": "EXCHANGE_POLICY",
            "content": "Regulated OTC desks and order-book exchanges must maintain at least 85% of total client digital assets in multi-signature air-gapped cold storage facilities. Hot wallet operational balances must not exceed the rolling 48-hour expected withdrawal volume calculated via Value-at-Risk models."
        }
    ]

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "financial_knowledge_db"),
        user=os.getenv("DB_USER", "crag_admin"),
        password=os.getenv("DB_PASSWORD", "crag_password"),
        port=5432
    )
    register_vector(conn)
    cursor = conn.cursor()

    print(f"Embedding and seeding {len(docs)} regulatory directives into pgvector...")
    for doc in docs:
        emb = embeddings.embed_query(doc["content"])
        cursor.execute("""
            INSERT INTO financial_documents (title, category, content, embedding)
            VALUES (%s, %s, %s, %s)
        """, (doc["title"], doc["category"], doc["content"], emb))

    conn.commit()
    cursor.close()
    conn.close()
    print("Seeding successfully completed.")

if __name__ == "__main__":
    seed_regulatory_corpus()
