"""Camera metadata preprocessing.

This script previously used `datasets.load_dataset` to stream the
`meta_Electronics.jsonl` metadata file. It now contains a manual JSONL
loader that supports:

- local file paths
- http(s) URLs (streaming)
- hf:// paths which are downloaded via `huggingface_hub.hf_hub_download`

Dependencies:
- requests
 - tqdm

Example:
    DATA_FILES = (
        "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Electronics.jsonl"
    )

Run the script to produce `data/items/cameras_meta.jsonl`.
"""

# %%
import json
import os
import sys
import time
from pprint import pprint
from typing import Any, Dict, Iterator
from urllib.parse import urlparse

import requests
from tqdm import tqdm

DATA_FILES = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Electronics.jsonl"


def is_camera(item: Dict[str, Any]) -> bool:
    """Check if an item is a camera based on its details."""
    return item.get("main_category") == "Camera & Photo"


def load_camera_metadata() -> Iterator[Dict[str, Any]]:
    """Yield camera items from the metadata source (streaming).

    This generator yields raw items whose `main_category` equals
    "Camera & Photo". It does not accumulate all items in memory.
    """
    print("Loading Electronics metadata...")

    # Support local files, http(s) URLs and hf:// paths
    def open_jsonl_source(source: str):
        """Yield parsed JSON objects from source.

        source can be:
        - local path: /path/to/file.jsonl
        - http(s) URL
        - hf://<repo>/<path> (will download the file using huggingface_hub)
        """
        parsed = urlparse(source)

        # hf:// scheme -> translate to huggingface.co resolve URL and stream
        if parsed.scheme == "hf":
            hf_path = source[len("hf://") :]
            parts = hf_path.split("/")
            if parts[0] == "datasets":
                if len(parts) < 4:
                    raise ValueError(f"Invalid hf:// dataset path: {source}")
                owner = parts[1]
                repo = parts[2]
                filename = "/".join(parts[3:])
            else:
                if len(parts) < 3:
                    raise ValueError(f"Invalid hf:// path: {source}")
                owner = parts[0]
                repo = parts[1]
                filename = "/".join(parts[2:])

            repo_id = f"{owner}/{repo}"
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            print(f"Streaming from Hugging Face URL: {url}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except Exception as e:
                        print(
                            f"Failed to parse JSON line: {e}", file=sys.stderr
                        )
            return

        raise FileNotFoundError(f"Could not open source: {source}")

    print("Filtering for cameras (streaming)...")
    for item in open_jsonl_source(DATA_FILES):
        try:
            if is_camera(item):
                yield item
        except Exception as e:
            print(f"Error processing item: {e}")


# %%
def clean_item(obj: Any) -> Any:
    """Recursively remove null/empty fields from JSON-like structures.

    Removes: None, empty strings, empty lists, empty dicts.
    Returns the cleaned object (may be None).
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return s if s != "" else None
    if isinstance(obj, list):
        cleaned = [clean_item(x) for x in obj]
        cleaned = [x for x in cleaned if x is not None]
        return cleaned if cleaned else None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cv = clean_item(v)
            if cv is not None:
                out[k] = cv
        return out if out else None
    return obj


OUTPUT_PATH = "data/items/cameras_meta.jsonl"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
count = 0
start_ts = time.time()
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    # Use tqdm to show a live progress bar: we don't know total, so leave total=None
    with tqdm(unit="items", desc="Processing", mininterval=1) as pbar:
        for camera in load_camera_metadata():
            cleaned = clean_item(camera)
            pbar.update(1)
            if not cleaned:
                continue
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            count += 1
            pbar.set_postfix({"written": count})

print(f"Wrote {count} cleaned camera items to {OUTPUT_PATH}")
