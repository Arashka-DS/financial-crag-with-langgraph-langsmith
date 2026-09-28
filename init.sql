CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS financial_documents (
    doc_id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    category VARCHAR(50), -- 'CBI_REGULATION', 'CRYPTO_TAX', 'EXCHANGE_POLICY'
    content TEXT,
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(1536), -- Standard OpenAI / text-embedding-3 dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_doc_embedding ON financial_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_doc_tsv ON financial_documents USING gin(tsv_content);
