import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def get_domain_paths(domain: str = "bicycle") -> tuple[str, str]:
    """Get the input and output paths for a given domain."""
    domain_lower = domain.lower().replace(" ", "_")
    input_path = f"data/items/{domain_lower}_meta_k50.jsonl"
    output_path = f"data/items/{domain_lower}_embedded_k50.jsonl"
    return input_path, output_path


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": False, "dtype": torch.float32},
    )


@st.cache_data(show_spinner=False)
def load_embeddings_file(embeddings_path: str) -> Tuple[np.ndarray, List[str]]:
    ids: List[str] = []
    embs: List[np.ndarray] = []
    with open(embeddings_path, "r") as infile:
        for line in infile:
            item = json.loads(line)
            emb_id = item.get("id", f"item_{len(ids)}")
            ids.append(emb_id)
            emb = np.array(item.get("embedding", []), dtype=np.float32)
            embs.append(emb)
    if not embs:
        return np.zeros((0, 0), dtype=np.float32), ids
    embs_np = np.asarray(embs, dtype=np.float32)
    return embs_np, ids


@st.cache_data(show_spinner=False)
def load_metadata_file(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    with open(metadata_path, "r") as infile:
        for i, line in enumerate(infile):
            item = json.loads(line)
            parent_asin = item.get("parent_asin")
            explicit_id = item.get("id")
            key = parent_asin or explicit_id or f"item_{i}"
            meta[key] = item
            if explicit_id and explicit_id != key:
                meta[explicit_id] = item
            if parent_asin and parent_asin != key:
                meta[parent_asin] = item
    return meta


def _encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    if hasattr(model, "encode_query"):
        return model.encode_query(text, convert_to_numpy=True, dtype=np.float32)
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def _encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if hasattr(model, "encode_corpus"):
        return model.encode_corpus(
            texts, convert_to_numpy=True, dtype=np.float32
        )
    return model.encode(texts, convert_to_numpy=True).astype(np.float32)


class ItemRetriever:
    def __init__(
        self,
        embeddings_path: str = None,
        metadata_path: str = None,
        model_name: str = MODEL_NAME,
        document_embedding: bool = False,
        domain: str = "bicycle",
    ):
        if embeddings_path is None or metadata_path is None:
            domain_input, domain_output = get_domain_paths(domain)
            if embeddings_path is None:
                embeddings_path = domain_output
            if metadata_path is None:
                metadata_path = domain_input

        if not document_embedding and not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found at {embeddings_path}"
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}"
            )

        self.model_name = model_name
        self.model = None  # we will fetch from cache when needed

        self.embeddings = None
        self.ids: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

        if not document_embedding:
            embs, ids = load_embeddings_file(embeddings_path)
            self.embeddings = embs  # float32 [N, D]
            self.ids = ids
            print(
                f"Loaded {len(self.ids)} embeddings with shape {self.embeddings.shape}"
            )

        self.metadata = load_metadata_file(metadata_path)
        print(f"Loaded metadata entries: {len(self.metadata)}")

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = get_model(self.model_name)
        return self.model

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        return _encode_texts(model, texts)

    def retrieve(self, query: str, top_k: int = 5):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Initialize with precomputed embeddings or call embed_batch first."
            )

        model = self._get_model()
        q = _encode_query(model, query)

        qn = q / (np.linalg.norm(q) + 1e-12)
        dn = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
        )
        sims = dn @ qn  # [N]

        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            emb_id = self.ids[i]
            meta = (
                self.metadata.get(emb_id)
                or self.metadata.get(str(emb_id))
                or {"id": emb_id}
            )
            results.append((meta, float(sims[i])))
        return results


def main(batch_size=1):
    domain = "running_shoes"

    INPUT_PATH = f"data/items/{domain}_meta.jsonl"
    OUTPUT_PATH = f"data/items/{domain}_embedded.jsonl"
    embedder = ItemRetriever(domain=domain, document_embedding=True)
    docs = []
    ids = []
    with open(INPUT_PATH, "r") as infile, open(OUTPUT_PATH, "w") as outfile:
        for line in tqdm(infile):
            item = json.loads(line)
            doc_id = item.get("parent_asin", f"item_{len(ids)}")
            content = item.get("content", "")

            if content:
                ids.append(doc_id)
                docs.append(content)

                if len(docs) == batch_size:
                    embeddings = embedder.embed_batch(docs)
                    for i, embedding in enumerate(embeddings):
                        result = {"id": ids[i], "embedding": embedding.tolist()}
                        outfile.write(json.dumps(result) + "\n")
                    docs = []
                    ids = []
        if docs:
            embeddings = embedder.embed_batch(docs)
            for i, embedding in enumerate(embeddings):
                result = {"id": ids[i], "embedding": embedding.tolist()}
                outfile.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
