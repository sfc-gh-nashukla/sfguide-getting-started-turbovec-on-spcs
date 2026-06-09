# Getting Started with TurboVec on SPCS

## 1. Log into Snowflake

Use the Snow CLI to connect to Snowflake.

```bash
snow connection test
```

## 2. Set up environment

```sql
USE ROLE ACCOUNTADMIN;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE SYSADMIN;
```

Create database, image repository, and stages:

```sql
USE ROLE SYSADMIN;
CREATE DATABASE IF NOT EXISTS TURBOVEC_DEMO;
USE DATABASE TURBOVEC_DEMO;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
CREATE IMAGE REPOSITORY TURBOVEC_DEMO.PUBLIC.TURBOVEC_REPO;
CREATE OR REPLACE STAGE YAML_STAGE;
CREATE OR REPLACE STAGE INDEXES ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

## 3. Set up compute pool

Create a CPU compute pool (no GPU needed — TurboVec uses SIMD kernels):

```sql
USE ROLE SYSADMIN;
CREATE COMPUTE POOL IF NOT EXISTS TURBOVEC_COMPUTE_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300;
```

Wait until the pool reaches `ACTIVE` or `IDLE`:

```sql
DESCRIBE COMPUTE POOL TURBOVEC_COMPUTE_POOL;
```

## 4. Build and push the Docker image

Build the TurboVec service image (linux/amd64 for SPCS):

```bash
docker build --rm --platform linux/amd64 \
  -t turbovec-spcs ./images/turbovec
```

Log in to the Snowflake Docker registry using Snow CLI (recommended — handles MFA):

```bash
snow spcs image-registry login
```

Tag and push:

```bash
docker tag turbovec-spcs \
  <ORG>-<ACCOUNT>.registry.snowflakecomputing.com/turbovec_demo/public/turbovec_repo/turbovec-spcs:latest

docker push \
  <ORG>-<ACCOUNT>.registry.snowflakecomputing.com/turbovec_demo/public/turbovec_repo/turbovec-spcs:latest
```

To find your registry URL:

```sql
SHOW IMAGE REPOSITORIES IN SCHEMA TURBOVEC_DEMO.PUBLIC;
```

## 5. Create the TurboVec service

Use an inline spec (no need to upload YAML to stage):

```sql
USE DATABASE TURBOVEC_DEMO;
USE SCHEMA PUBLIC;

CREATE SERVICE IF NOT EXISTS TURBOVEC
  IN COMPUTE POOL TURBOVEC_COMPUTE_POOL
  FROM SPECIFICATION $$
spec:
  containers:
    - name: turbovec
      image: <ORG>-<ACCOUNT>.registry.snowflakecomputing.com/turbovec_demo/public/turbovec_repo/turbovec-spcs:latest
      env:
        TURBOVEC_DIM: "1536"
        TURBOVEC_BIT_WIDTH: "4"
        TURBOVEC_INDEX_PATH: "/data/turbovec_index.tvim"
      resources:
        requests:
          memory: 2Gi
          cpu: 2
        limits:
          memory: 4Gi
          cpu: 4
      readinessProbe:
        port: 8000
        path: /health
      volumeMounts:
        - name: index-storage
          mountPath: /data
  endpoints:
    - name: api
      port: 8000
      public: false
  volumes:
    - name: index-storage
      source: local
$$
MIN_INSTANCES = 1
MAX_INSTANCES = 1;
```

Verify it's running:

```sql
SELECT SYSTEM$GET_SERVICE_STATUS('TURBOVEC_DEMO.PUBLIC.TURBOVEC');
```

Expected output: `"status":"READY","message":"Running"`

## 6. Create service functions

SPCS service functions wrap data in a specific format. The TurboVec service exposes `/sf/add` and `/sf/search` endpoints for this:

```sql
CREATE OR REPLACE FUNCTION turbovec_sf_add(input OBJECT)
RETURNS VARIANT
SERVICE = TURBOVEC
ENDPOINT = api
MAX_BATCH_ROWS = 1
AS '/sf/add';

CREATE OR REPLACE FUNCTION turbovec_sf_search(input OBJECT)
RETURNS VARIANT
SERVICE = TURBOVEC
ENDPOINT = api
MAX_BATCH_ROWS = 1
AS '/sf/search';
```

## 7. Load the benchmark dataset

We use the public [Qdrant/DBpedia OpenAI 1536-dim](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) dataset — 10K pre-embedded Wikipedia entities.

Download and export to parquet (locally):

```bash
pip install datasets pyarrow numpy
python3 experiments/export_dbpedia.py
```

Upload to Snowflake stage:

```bash
snow sql -q "PUT file://~/data/py-turboquant/export/dbpedia_10k_with_text.parquet @TURBOVEC_DEMO.PUBLIC.YAML_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
```

Load into a Snowflake table (for native vector search comparison):

```sql
CREATE OR REPLACE FILE FORMAT parquet_format TYPE = PARQUET;

CREATE OR REPLACE TABLE dbpedia_with_text AS
SELECT 
  $1:id::INTEGER AS id,
  $1:text::VARCHAR AS content,
  $1:embedding::VECTOR(FLOAT, 1536) AS embedding
FROM @YAML_STAGE/dbpedia_10k_with_text.parquet (FILE_FORMAT => 'parquet_format');
```

Load into TurboVec (batch 100 vectors at a time due to service function payload limits):

```sql
CREATE OR REPLACE PROCEDURE load_vectors_batch(start_id INT, end_id INT)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
  LET batch_size INT := 100;
  LET current_start INT := :start_id;
  LET loaded INT := 0;
  WHILE (current_start < :end_id) DO
    LET current_end INT := LEAST(current_start + batch_size, :end_id);
    SELECT turbovec_sf_add(OBJECT_CONSTRUCT(
      'vectors', ARRAY_AGG(embedding::ARRAY),
      'ids', ARRAY_AGG(id)
    ))
    FROM dbpedia_with_text
    WHERE id >= :current_start AND id < :current_end;
    loaded := loaded + (current_end - current_start);
    current_start := current_end;
  END WHILE;
  RETURN 'Loaded ' || loaded || ' vectors';
END;
$$;

-- Load all 10K vectors (~2 minutes)
CALL load_vectors_batch(0, 10000);
```

## 8. Run the benchmark

### Snowflake Native Vector Search (exact brute-force):

```sql
SELECT 
  d.id,
  VECTOR_COSINE_SIMILARITY(d.embedding, q.embedding) AS score
FROM dbpedia_with_text d, dbpedia_queries q
WHERE q.id = 0
ORDER BY score DESC
LIMIT 5;
```

### TurboVec (4-bit quantized, on SPCS):

```sql
SELECT turbovec_sf_search(OBJECT_CONSTRUCT(
  'query', q.embedding::ARRAY,
  'k', 5
)):ids AS top5_ids,
turbovec_sf_search(OBJECT_CONSTRUCT(
  'query', q.embedding::ARRAY,
  'k', 5
)):latency_ms AS latency_ms
FROM dbpedia_queries q
WHERE q.id = 0;
```

### Cortex Search (hybrid BM25 + semantic):

```sql
CREATE OR REPLACE CORTEX SEARCH SERVICE DBPEDIA_SEARCH
  ON content
  WAREHOUSE = COMPUTE_WH
  TARGET_LAG = '1 hour'
  AS (SELECT id, content FROM dbpedia_with_text);

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'TURBOVEC_DEMO.PUBLIC.DBPEDIA_SEARCH',
    '{"query": "your query text here", "columns": ["id", "content"], "limit": 5}'
  )
):results;
```

## 9. Expected results

Benchmark on 100K vectors from Qdrant/DBpedia (d=1536, OpenAI text-embedding-3-large):

| Method | Recall@5 | Search Latency | Memory (100K vecs) |
|--------|----------|----------------|-------------------|
| Snowflake Native (FP32 exact) | 1.000 | ~500ms (warehouse) | 585.9 MB |
| **TurboVec 4-bit (SPCS)** | **1.000** | **13ms** | **73.6 MB** |

TurboVec achieves identical recall to exact brute-force search with 8x less memory and ~40x lower latency.

## 10. Multi-tenant filtered search

```sql
-- Add vectors with tenant assignment
SELECT turbovec_sf_add(OBJECT_CONSTRUCT(
  'vectors', ARRAY_AGG(embedding::ARRAY),
  'ids', ARRAY_AGG(id),
  'tenant_id', 'tenant_a'
))
FROM dbpedia_with_text
WHERE id < 100;

-- Search restricted to tenant_a only
SELECT turbovec_sf_search(OBJECT_CONSTRUCT(
  'query', (SELECT embedding::ARRAY FROM dbpedia_queries WHERE id = 0),
  'k', 5,
  'tenant_id', 'tenant_a'
));
```

## 11. View telemetry

```sql
-- Check service logs
CALL SYSTEM$GET_SERVICE_LOGS('TURBOVEC_DEMO.PUBLIC.TURBOVEC', '0', 'turbovec', 20);
```

## Suspend and resume

```sql
ALTER SERVICE TURBOVEC SUSPEND;  -- stop billing
ALTER SERVICE TURBOVEC RESUME;   -- restart
```

## Cleanup

```sql
USE ROLE SYSADMIN;
DROP SERVICE IF EXISTS TURBOVEC_DEMO.PUBLIC.TURBOVEC;
DROP SERVICE IF EXISTS TURBOVEC_DEMO.PUBLIC.DBPEDIA_SEARCH;
DROP COMPUTE POOL IF EXISTS TURBOVEC_COMPUTE_POOL;
DROP DATABASE IF EXISTS TURBOVEC_DEMO;
```
