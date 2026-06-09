/*
 * TurboVec on SPCS - Create Services
 * Run after Docker images are pushed and spec files are uploaded
 */

USE ROLE SYSADMIN;
USE DATABASE TURBOVEC_DEMO;
USE SCHEMA PUBLIC;

-- ============================================================
-- 1. CREATE TURBOVEC SERVICE
-- ============================================================
CREATE SERVICE IF NOT EXISTS TURBOVEC
  IN COMPUTE POOL TURBOVEC_COMPUTE_POOL
  FROM @YAML_STAGE
  SPEC = 'turbovec.yaml'
  MIN_INSTANCES = 1
  MAX_INSTANCES = 3;

-- ============================================================
-- 2. VERIFY SERVICE STATUS
-- ============================================================
SELECT SYSTEM$GET_SERVICE_STATUS('TURBOVEC_DEMO.PUBLIC.TURBOVEC');
SHOW ENDPOINTS IN SERVICE TURBOVEC_DEMO.PUBLIC.TURBOVEC;

-- ============================================================
-- 3. CREATE SERVICE FUNCTION (for SQL access)
-- ============================================================
CREATE OR REPLACE FUNCTION turbovec_add(vectors ARRAY, ids ARRAY, tenant_id VARCHAR)
RETURNS VARIANT
SERVICE = TURBOVEC_DEMO.PUBLIC.TURBOVEC
ENDPOINT = api
AS '/add';

CREATE OR REPLACE FUNCTION turbovec_search(query ARRAY, k INTEGER, tenant_id VARCHAR)
RETURNS VARIANT
SERVICE = TURBOVEC_DEMO.PUBLIC.TURBOVEC
ENDPOINT = api
AS '/search';

CREATE OR REPLACE FUNCTION turbovec_health()
RETURNS VARIANT
SERVICE = TURBOVEC_DEMO.PUBLIC.TURBOVEC
ENDPOINT = api
AS '/health';

-- ============================================================
-- 4. GRANT ACCESS
-- ============================================================
USE ROLE SECURITYADMIN;
GRANT USAGE ON SERVICE TURBOVEC_DEMO.PUBLIC.TURBOVEC TO ROLE TURBOVEC_ROLE;

-- ============================================================
-- 5. TEST
-- ============================================================
USE ROLE SYSADMIN;
SELECT turbovec_health();

-- ============================================================
-- 6. SAMPLE: Embed and search using Cortex AI + TurboVec
-- ============================================================
/*
-- Step 1: Create a documents table
CREATE OR REPLACE TABLE documents (
    id INTEGER,
    tenant_id VARCHAR,
    content VARCHAR,
    embedding VECTOR(FLOAT, 768)
);

-- Step 2: Insert documents with embeddings
INSERT INTO documents (id, tenant_id, content, embedding)
SELECT
    seq4() AS id,
    'tenant_a' AS tenant_id,
    content,
    AI_EMBED('snowflake-arctic-embed-m-v1.5', content) AS embedding
FROM (
    SELECT 'TurboVec uses data-oblivious quantization for vector compression' AS content
    UNION ALL
    SELECT 'Snowpark Container Services enables custom containers in Snowflake'
    UNION ALL
    SELECT 'RAG systems combine retrieval with LLM generation for grounded answers'
);

-- Step 3: Search via service function
SELECT turbovec_search(
    AI_EMBED('snowflake-arctic-embed-m-v1.5', 'How does vector compression work?')::ARRAY,
    5,
    'tenant_a'
);
*/
