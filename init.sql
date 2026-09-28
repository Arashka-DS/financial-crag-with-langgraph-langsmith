CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS regulatory_docs (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(100) NOT NULL UNIQUE,
    title VARCHAR(255),
    document_text TEXT NOT NULL,
    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', document_text)) STORED,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lexical Full-Text Search GIN Index
CREATE INDEX IF NOT EXISTS idx_regulatory_docs_fts ON regulatory_docs USING gin(fts_vector);

-- Dense Vector HNSW Cosine Index
CREATE INDEX IF NOT EXISTS idx_regulatory_docs_hnsw ON regulatory_docs USING hnsw (embedding vector_cosine_ops);
