#!/usr/bin/env python3
"""Cluster item embeddings and sample one representative per cluster.

Reads an embeddings JSONL (records with 'id' and 'embedding') and a metadata
JSONL (original items). Performs k-means clustering (k default 50) and picks
the item closest to each centroid as the cluster representative. Writes a
curated JSONL with the selected metadata items and a small report.

Usage:
    python scripts/cluster_and_sample.py \
        --emb data/items/digital_camera_embedded.jsonl \
        --meta data/items/digital_camera_meta.jsonl \
        --k 50

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def load_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    ids: List[str] = []
    embs: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: failed to parse JSON on {path}:{i}")
                continue
            eid = obj.get("id") or f"item_{len(ids)}"
            emb = obj.get("embedding")
            if emb is None:
                continue
            ids.append(eid)
            embs.append(emb)
    if not embs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.asarray(embs, dtype=np.float32), ids


def load_metadata_map(meta_path: Path) -> Dict[str, Dict]:
    meta: Dict[str, Dict] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: failed to parse JSON on {meta_path}:{i}")
                continue
            key = obj.get("parent_asin") or obj.get("id") or f"item_{i}"
            meta[key] = obj
    return meta


def cluster_and_pick(
    embs: np.ndarray, ids: List[str], k: int = 50
) -> List[int]:
    """Return indices (into ids/embs) of selected representatives."""
    n = embs.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    if HAVE_SKLEARN:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(embs)
        centroids = km.cluster_centers_
    else:
        # Simple kmeans-like: initialize random centroids and do a few iterations
        rng = np.random.default_rng(42)
        centroids = embs[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(10):
            # assign
            dists = np.linalg.norm(
                embs[:, None, :] - centroids[None, :, :], axis=2
            )
            labels = np.argmin(dists, axis=1)
            # update
            for j in range(k):
                members = embs[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)

    selected_idx: List[int] = []
    for j in range(centroids.shape[0]):
        members_idx = np.where(labels == j)[0]
        if members_idx.size == 0:
            continue
        member_embs = embs[members_idx]
        dists = np.linalg.norm(member_embs - centroids[j], axis=1)
        chosen = members_idx[int(np.argmin(dists))]
        selected_idx.append(int(chosen))

    return selected_idx


def write_curated(
    output_path: Path, selected_ids: List[str], meta_map: Dict[str, Dict]
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for sid in selected_ids:
            obj = meta_map.get(sid) or meta_map.get(str(sid))
            if obj is None:
                # If metadata missing, write a minimal record
                obj = {"id": sid}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="Embeddings JSONL path")
    parser.add_argument("--meta", required=True, help="Metadata JSONL path")
    parser.add_argument(
        "--k", type=int, default=50, help="Number of clusters / items to sample"
    )
    parser.add_argument("--out", default=None, help="Output curated JSONL path")
    args = parser.parse_args(argv[1:])

    emb_path = Path(args.emb)
    meta_path = Path(args.meta)
    out_path = (
        Path(args.out)
        if args.out
        else meta_path.with_name(
            meta_path.stem + f"_curated_k{args.k}" + meta_path.suffix
        )
    )

    embs, ids = load_embeddings(emb_path)
    meta_map = load_metadata_map(meta_path)

    if embs.size == 0:
        print("No embeddings found. Exiting.")
        return 1

    sel_idx = cluster_and_pick(embs, ids, k=args.k)
    selected_ids = [ids[i] for i in sel_idx]

    write_curated(out_path, selected_ids, meta_map)

    # write a small report
    rpt = out_path.with_suffix(".report.txt")
    with rpt.open("w", encoding="utf-8") as f:
        f.write(f"embeddings: {emb_path}\n")
        f.write(f"metadata: {meta_path}\n")
        f.write(f"k: {args.k}\n")
        f.write(f"available_embeddings: {len(ids)}\n")
        f.write(f"selected: {len(selected_ids)}\n")

    print(
        f"Wrote curated file: {out_path} (selected {len(selected_ids)} items). Report: {rpt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
