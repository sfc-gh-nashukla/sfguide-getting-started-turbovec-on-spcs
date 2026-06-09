"""
TurboVec SPCS Service — FastAPI vector search service for Snowpark Container Services.

Deploys TurboVec as a containerized REST API on Snowflake compute pools.
Supports multi-tenant filtered search with per-tenant telemetry.
"""
import os
import time
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from turbovec import IdMapIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("turbovec-spcs")

INDEX_STATE = {}


class IndexConfig(BaseModel):
    dim: int = 1536
    bit_width: int = 4


class AddVectorsRequest(BaseModel):
    vectors: list[list[float]]
    ids: list[int]
    tenant_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: list[float]
    k: int = 10
    tenant_id: Optional[str] = None
    allowlist: Optional[list[int]] = None


class SearchResponse(BaseModel):
    scores: list[float]
    ids: list[int]
    latency_ms: float
    vectors_in_index: int
    tenant_id: Optional[str] = None


class TelemetryRecord(BaseModel):
    tenant_id: Optional[str]
    query_id: str
    latency_us: int
    k_returned: int
    allowlist_size: Optional[int]
    index_memory_bytes: int
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = IndexConfig(
        dim=int(os.getenv("TURBOVEC_DIM", "1536")),
        bit_width=int(os.getenv("TURBOVEC_BIT_WIDTH", "4")),
    )
    INDEX_STATE["config"] = config
    INDEX_STATE["index"] = IdMapIndex(dim=config.dim, bit_width=config.bit_width)
    INDEX_STATE["tenant_ids"] = {}
    INDEX_STATE["telemetry"] = []
    INDEX_STATE["total_vectors"] = 0
    logger.info(f"TurboVec SPCS initialized: dim={config.dim}, bit_width={config.bit_width}")
    yield
    logger.info("TurboVec SPCS shutting down")


app = FastAPI(
    title="TurboVec SPCS Service",
    description="Data-oblivious vector search on Snowpark Container Services",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "vectors": INDEX_STATE.get("total_vectors", 0),
        "config": INDEX_STATE.get("config", {}).dict() if INDEX_STATE.get("config") else {},
    }


@app.post("/health")
def health_post(request: dict = None):
    """SPCS service functions send POST. Handle both GET and POST."""
    data = {"status": "healthy", "vectors": INDEX_STATE.get("total_vectors", 0)}
    if request and "data" in request:
        return {"data": [[0, data]]}
    return data


@app.post("/sf/add")
def sf_add_vectors(request: dict):
    """SPCS service function wrapper for /add."""
    row = request["data"][0]
    payload = row[1] if len(row) > 1 else row[0]
    req = AddVectorsRequest(**payload)
    result = add_vectors(req)
    return {"data": [[0, result]]}


@app.post("/sf/search")
def sf_search(request: dict):
    """SPCS service function wrapper for /search."""
    row = request["data"][0]
    payload = row[1] if len(row) > 1 else row[0]
    req = SearchRequest(**payload)
    result = search(req)
    return {"data": [[0, result.dict()]]}


@app.post("/add")
def add_vectors(request: AddVectorsRequest):
    index = INDEX_STATE["index"]
    vectors = np.array(request.vectors, dtype=np.float32)
    ids = np.array(request.ids, dtype=np.uint64)

    if vectors.shape[1] != INDEX_STATE["config"].dim:
        raise HTTPException(400, f"Expected dim={INDEX_STATE['config'].dim}, got {vectors.shape[1]}")

    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    index.add_with_ids(vectors, ids)

    if request.tenant_id:
        if request.tenant_id not in INDEX_STATE["tenant_ids"]:
            INDEX_STATE["tenant_ids"][request.tenant_id] = set()
        INDEX_STATE["tenant_ids"][request.tenant_id].update(int(i) for i in ids)

    INDEX_STATE["total_vectors"] += len(ids)

    return {"added": len(ids), "total_vectors": INDEX_STATE["total_vectors"]}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    index = INDEX_STATE["index"]
    query = np.array([request.query], dtype=np.float32)
    query /= np.linalg.norm(query, axis=-1, keepdims=True)

    allowlist = None
    allowlist_size = None

    if request.allowlist:
        allowlist = np.array(request.allowlist, dtype=np.uint64)
        allowlist_size = len(allowlist)
    elif request.tenant_id and request.tenant_id in INDEX_STATE["tenant_ids"]:
        tenant_ids = list(INDEX_STATE["tenant_ids"][request.tenant_id])
        allowlist = np.array(tenant_ids, dtype=np.uint64)
        allowlist_size = len(allowlist)

    t0 = time.perf_counter()
    if allowlist is not None:
        scores, ids = index.search(query, k=request.k, allowlist=allowlist)
    else:
        scores, ids = index.search(query, k=request.k)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000

    telemetry = {
        "tenant_id": request.tenant_id,
        "latency_us": int((t1 - t0) * 1_000_000),
        "k_returned": len(ids[0]),
        "allowlist_size": allowlist_size,
        "index_memory_bytes": INDEX_STATE["total_vectors"] * INDEX_STATE["config"].dim * INDEX_STATE["config"].bit_width // 8,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    INDEX_STATE["telemetry"].append(telemetry)

    return SearchResponse(
        scores=[float(s) for s in scores[0]],
        ids=[int(i) for i in ids[0]],
        latency_ms=round(latency_ms, 3),
        vectors_in_index=INDEX_STATE["total_vectors"],
        tenant_id=request.tenant_id,
    )


@app.get("/telemetry")
def get_telemetry(limit: int = 100):
    return INDEX_STATE["telemetry"][-limit:]


@app.post("/persist")
def persist_index():
    path = os.getenv("TURBOVEC_INDEX_PATH", "/data/turbovec_index.tvim")
    INDEX_STATE["index"].write(path)
    return {"persisted_to": path, "vectors": INDEX_STATE["total_vectors"]}


@app.post("/load")
def load_index():
    path = os.getenv("TURBOVEC_INDEX_PATH", "/data/turbovec_index.tvim")
    if not os.path.exists(path):
        raise HTTPException(404, f"No index at {path}")
    INDEX_STATE["index"] = IdMapIndex.load(path)
    return {"loaded_from": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
