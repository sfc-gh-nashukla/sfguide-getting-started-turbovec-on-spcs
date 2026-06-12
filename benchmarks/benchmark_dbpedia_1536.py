#!/usr/bin/env python3
"""
Reproducible benchmark: TurboVec vs FAISS on Qdrant/DBpedia OpenAI 1536-dim.

Dataset: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M (HuggingFace)
Vectors: 100K database, 1000 queries (pre-embedded by OpenAI text-embedding-3-large)

Anyone can reproduce:
  pip install turbovec faiss-cpu numpy datasets
  python3 benchmark_dbpedia_1536.py
"""
import os
import sys
import json
import time
import numpy as np

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

DIM = 1536
BIT_WIDTH = 4
N_DATABASE = 100000
N_QUERIES = 1000
K_VALUES = [1, 5, 10, 50]
SEED = 42

def load_data():
    path = os.path.join(DATA_DIR, "openai-1536.npy")
    export_dir = os.path.join(DATA_DIR, "export")

    if os.path.exists(path) and os.path.getsize(path) > 500_000_000:
        all_vecs = np.load(path)
    elif os.path.exists(os.path.join(export_dir, "dbpedia_shard_00.parquet")):
        import pyarrow.parquet as pq
        print("Loading from parquet shards...")
        shards = sorted([f for f in os.listdir(export_dir) if f.startswith("dbpedia_shard_")])
        all_embeddings = []
        for shard_file in shards:
            table = pq.read_table(os.path.join(export_dir, shard_file))
            all_embeddings.extend(table["embedding"].to_pylist())
        all_vecs = np.array(all_embeddings, dtype=np.float32)
        np.save(path, all_vecs)
        print(f"Cached to {path}")
    else:
        print(f"Dataset not found. Downloading 101K vectors from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset(
            "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
            split="train[:101000]"
        )
        all_embeddings = ds["text-embedding-3-large-1536-embedding"]
        all_vecs = np.array(all_embeddings, dtype=np.float32)
        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(path, all_vecs)

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:N_DATABASE]].astype(np.float32)
    queries = all_vecs[idx[N_DATABASE:N_DATABASE + N_QUERIES]].astype(np.float32)
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return database, queries


def compute_ground_truth(queries, database, k):
    scores = queries @ database.T
    return np.argsort(-scores, axis=1)[:, :k]


def recall_at_k(true_topk, predicted_topk, k):
    hits = 0
    for i in range(len(true_topk)):
        hits += len(set(true_topk[i, :k]) & set(predicted_topk[i, :k]))
    return hits / (len(true_topk) * k)


def benchmark_turbovec(database, queries, ground_truth):
    from turbovec import TurboQuantIndex

    index = TurboQuantIndex(DIM, bit_width=BIT_WIDTH)

    t0 = time.perf_counter()
    index.add(database)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores, indices = index.search(queries, k=max(K_VALUES))
    search_time = time.perf_counter() - t0

    indices = np.array(indices)
    recalls = {}
    for k in K_VALUES:
        recalls[k] = round(recall_at_k(ground_truth[:, :k], indices[:, :k], k), 4)

    return {
        "method": f"TurboVec {BIT_WIDTH}-bit",
        "build_time_sec": round(build_time, 3),
        "search_time_sec": round(search_time, 4),
        "ms_per_query": round(search_time / N_QUERIES * 1000, 3),
        "qps": round(N_QUERIES / search_time, 1),
        "recalls": {f"R@{k}": v for k, v in recalls.items()},
        "memory_mb": round(N_DATABASE * DIM * BIT_WIDTH / 8 / (1024**2), 2),
    }


def benchmark_faiss(database, queries, ground_truth):
    import faiss

    m = DIM // 2
    nbits = 8
    index = faiss.IndexPQ(DIM, m, nbits, faiss.METRIC_INNER_PRODUCT)

    t0 = time.perf_counter()
    index.train(database)
    index.add(database)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, indices = index.search(queries, max(K_VALUES))
    search_time = time.perf_counter() - t0

    recalls = {}
    for k in K_VALUES:
        recalls[k] = round(recall_at_k(ground_truth[:, :k], indices[:, :k], k), 4)

    return {
        "method": "FAISS PQ (m=768, nbits=8)",
        "build_time_sec": round(build_time, 3),
        "search_time_sec": round(search_time, 4),
        "ms_per_query": round(search_time / N_QUERIES * 1000, 3),
        "qps": round(N_QUERIES / search_time, 1),
        "recalls": {f"R@{k}": v for k, v in recalls.items()},
        "memory_mb": round(N_DATABASE * DIM * nbits / 8 / (1024**2), 2),
    }


def benchmark_exact(database, queries, ground_truth):
    t0 = time.perf_counter()
    scores = queries @ database.T
    indices = np.argsort(-scores, axis=1)[:, :max(K_VALUES)]
    search_time = time.perf_counter() - t0

    return {
        "method": "Exact (FP32 brute force)",
        "build_time_sec": 0,
        "search_time_sec": round(search_time, 4),
        "ms_per_query": round(search_time / N_QUERIES * 1000, 3),
        "qps": round(N_QUERIES / search_time, 1),
        "recalls": {f"R@{k}": 1.0 for k in K_VALUES},
        "memory_mb": round(N_DATABASE * DIM * 4 / (1024**2), 2),
    }


def main():
    print("=" * 70)
    print("BENCHMARK: TurboVec vs FAISS vs Exact")
    print(f"Dataset: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M")
    print(f"Vectors: {N_DATABASE} database, {N_QUERIES} queries, dim={DIM}")
    print(f"Bit width: {BIT_WIDTH}, seed: {SEED}")
    print("=" * 70)

    database, queries = load_data()
    print(f"\nData loaded: database={database.shape}, queries={queries.shape}")

    print("\nComputing ground truth (exact top-k)...")
    ground_truth = compute_ground_truth(queries, database, max(K_VALUES))

    print("\n--- Exact (FP32) ---")
    exact_result = benchmark_exact(database, queries, ground_truth)
    print(f"  {exact_result['ms_per_query']}ms/query, {exact_result['memory_mb']}MB")

    print("\n--- TurboVec 4-bit ---")
    tv_result = benchmark_turbovec(database, queries, ground_truth)
    print(f"  Build: {tv_result['build_time_sec']}s")
    print(f"  Search: {tv_result['ms_per_query']}ms/query ({tv_result['qps']} QPS)")
    print(f"  Memory: {tv_result['memory_mb']}MB")
    print(f"  Recall: {tv_result['recalls']}")

    print("\n--- FAISS PQ ---")
    faiss_result = benchmark_faiss(database, queries, ground_truth)
    print(f"  Build: {faiss_result['build_time_sec']}s")
    print(f"  Search: {faiss_result['ms_per_query']}ms/query ({faiss_result['qps']} QPS)")
    print(f"  Memory: {faiss_result['memory_mb']}MB")
    print(f"  Recall: {faiss_result['recalls']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'ms/q':<8} {'Memory':<10}")
    print("-" * 70)
    for r in [exact_result, tv_result, faiss_result]:
        print(f"{r['method']:<30} {r['recalls']['R@1']:<8} {r['recalls']['R@5']:<8} "
              f"{r['recalls']['R@10']:<8} {r['ms_per_query']:<8} {r['memory_mb']}MB")

    all_results = {
        "dataset": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "dataset_url": "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "n_database": N_DATABASE,
        "n_queries": N_QUERIES,
        "dim": DIM,
        "seed": SEED,
        "results": [exact_result, tv_result, faiss_result],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark_dbpedia_1536.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"\nTo reproduce: pip install turbovec faiss-cpu numpy datasets")
    print(f"              python3 {os.path.basename(__file__)}")


if __name__ == "__main__":
    main()
