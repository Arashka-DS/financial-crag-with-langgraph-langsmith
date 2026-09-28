FROM python:3.11-slim

WORKDIR /app

# libpq-dev is required for psycopg2 connection to PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# The execution command is overridden by docker-compose to handle seeding
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
