import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

INPUT_PATH = "data/items/bikes_meta.jsonl"
OUTPUT_PATH = "data/items/bikes_embedded.jsonl"


class ItemRetriever:
    def __init__(
        self,
        embeddings_path=OUTPUT_PATH,
        metadata_path=INPUT_PATH,
        model_name=MODEL_NAME,
        document_embedding=False,
    ):
        # Validate paths depending on mode
        if not document_embedding and not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found at {embeddings_path}"
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}"
            )

        # Model is created lazily to avoid unnecessary startup cost when only
        # creating the object for embedding generation (document_embedding=True)
        self.model_name = model_name
        self.model = None

        # Embeddings will be None if we're in document_embedding mode (we'll compute them)
        self.embeddings = None
        self.ids = []
        self.metadata = {}

        # Load embeddings only when not in document_embedding mode
        if not document_embedding:
            # Load embeddings with proper dtype handling
            print("Loading embeddings from", embeddings_path)
            embeddings_list = []
            with open(embeddings_path, "r") as infile:
                for line in infile:
                    item = json.loads(line)
                    # Use the id field from the embeddings file
                    emb_id = item.get("id")
                    if emb_id is None:
                        # fallback to generated id
                        emb_id = f"item_{len(self.ids)}"
                    self.ids.append(emb_id)
                    # Ensure consistent float32 dtype
                    embedding = np.array(
                        item.get("embedding", []), dtype=np.float32
                    )
                    embeddings_list.append(embedding)

            # Convert to numpy array with consistent dtype
            if embeddings_list:
                self.embeddings = np.array(embeddings_list, dtype=np.float32)
                print(
                    f"Loaded {len(self.embeddings)} embeddings with shape {self.embeddings.shape}"
                )
            else:
                self.embeddings = np.array([], dtype=np.float32)
                print("No embeddings found in embeddings file.")

        # Load metadata and index by plausible ids (parent_asin and any explicit id)
        print("Loading metadata from", metadata_path)
        with open(metadata_path, "r") as infile:
            for i, line in enumerate(infile):
                item = json.loads(line)
                # Primary id candidates
                parent_asin = item.get("parent_asin")
                explicit_id = item.get("id")
                key = parent_asin or explicit_id or f"item_{i}"

                # Store metadata for both parent_asin and explicit id if available
                self.metadata[key] = item
                if explicit_id and explicit_id != key:
                    self.metadata[explicit_id] = item
                if parent_asin and parent_asin != key:
                    self.metadata[parent_asin] = item

    def embed_batch(self, texts):
        # lazy-load model
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model.encode(texts, convert_to_numpy=True, dtype=np.float32)

    def retrieve(self, query, top_k=5):
        # Encode query with consistent dtype
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Initialize ItemRetriever with precomputed embeddings or call embed_batch to generate embeddings first."
            )

        # lazy-load model if needed for query encoding
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        query_embedding = self.model.encode_query(
            query, convert_to_numpy=True, dtype=np.float32
        )

        # Calculate similarities using numpy for better performance
        # Normalize embeddings for cosine similarity
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            raise ValueError("Query embedding has zero norm")
        query_norm = query_embedding / q_norm

        embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_indices:
            emb_id = self.ids[i]
            metadata = self.metadata.get(emb_id)
            # fallback to any mapping if direct id lookup fails
            if metadata is None:
                # try parent_asin mapping or use a basic placeholder
                metadata = self.metadata.get(str(emb_id), {"id": emb_id})
            results.append((metadata, float(similarities[i])))

        return results


def main(batch_size=8):
    embedder = ItemRetriever(document_embedding=True)
    docs = []
    ids = []
    with open(INPUT_PATH, "r") as infile, open(OUTPUT_PATH, "w") as outfile:
        for line in tqdm(infile):
            item = json.loads(line)
            # Use parent_asin as the document ID and content field as the document
            doc_id = item.get("parent_asin", f"item_{len(ids)}")
            content = item.get("content", "")

            if content:  # Only process items with content
                ids.append(doc_id)
                docs.append(content)

                if len(docs) == batch_size:
                    embeddings = embedder.embed_batch(docs)
                    for i, embedding in enumerate(embeddings):
                        result = {"id": ids[i], "embedding": embedding.tolist()}
                        outfile.write(json.dumps(result) + "\n")
                    docs = []
                    ids = []
        # process any remaining docs
        if docs:
            embeddings = embedder.embed_batch(docs)
            for i, embedding in enumerate(embeddings):
                result = {"id": ids[i], "embedding": embedding.tolist()}
                outfile.write(json.dumps(result) + "\n")


# Example usage
if __name__ == "__main__":
    main()
    # retriever = ItemRetriever()

    # query = "What is the capital of China?"
    # print(f"Retrieving items for query: {query}")
    # results = retriever.retrieve(query)

    # print("Top retrieved items:")
    # for metadata, score in results:
    #     print(f"Metadata: {metadata}, Similarity Score: {score:.4f}")
