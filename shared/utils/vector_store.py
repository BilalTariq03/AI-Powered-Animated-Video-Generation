"""
shared/utils/vector_store.py
─────────────────────────────
Persistent memory layer using ChromaDB.
Falls back to a no-op stub if chromadb is not installed so that
Phase 2 / Phase 3 can run without the full dependency set.

A lightweight hash-based EmbeddingFunction is injected so ChromaDB
never loads its default ONNX model (which requires ~5 GB of RAM).
"""

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from config import CHROMA_PATH, COLLECTION

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb import EmbeddingFunction, Embeddings
    import numpy as np
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class _HashEmbeddingFunction(EmbeddingFunction if _CHROMA_AVAILABLE else object):
    """
    Zero-dependency embedding via character trigram hashing.
    Avoids loading the ONNX sentence-transformer model entirely.
    Semantic quality is low but the store is used as a key-value
    cache here, so exact-key lookup is what matters.
    """
    DIM = 128

    def __call__(self, input: list) -> list:
        results = []
        for text in input:
            vec = [0.0] * self.DIM
            text = str(text)
            for i in range(max(1, len(text) - 2)):
                trigram = text[i:i + 3]
                idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results


class _StubCollection:
    """No-op ChromaDB collection used when chromadb is not installed."""
    def get(self, **_):    return {"ids": [], "documents": [], "metadatas": []}
    def add(self, **_):    pass
    def update(self, **_): pass
    def query(self, **_):  return {"documents": [[]], "metadatas": [[]]}
    def count(self):       return 0


class VectorMemory:
    def __init__(self):
        if _CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(
                    path=CHROMA_PATH,
                    settings=Settings(anonymized_telemetry=False),
                )
                ef = _HashEmbeddingFunction()
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=COLLECTION,
                        embedding_function=ef,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception:
                    # Existing collection used a different embedding function — recreate it
                    self.client.delete_collection(name=COLLECTION)
                    self.collection = self.client.get_or_create_collection(
                        name=COLLECTION,
                        embedding_function=ef,
                        metadata={"hnsw:space": "cosine"},
                    )
            except Exception as e:
                print(f"[VectorMemory] ChromaDB init failed ({e}), using stub.")
                self.client     = None
                self.collection = _StubCollection()
        else:
            self.client     = None
            self.collection = _StubCollection()

    def store(self, key: str, data: Any, metadata: Optional[Dict] = None) -> str:
        doc_id  = str(uuid.uuid4())
        content = json.dumps(data) if not isinstance(data, str) else data
        meta    = {**(metadata or {}), "key": key}

        existing = self.collection.get(where={"key": key})
        if existing["ids"]:
            self.collection.update(ids=existing["ids"], documents=[content], metadatas=[meta])
            return existing["ids"][0]

        self.collection.add(ids=[doc_id], documents=[content], metadatas=[meta])
        return doc_id

    def retrieve(self, key: str) -> Optional[Any]:
        results = self.collection.get(where={"key": key})
        if not results["documents"]:
            return None
        try:
            return json.loads(results["documents"][0])
        except json.JSONDecodeError:
            return results["documents"][0]

    def query(self, text: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[text],
            n_results=min(n_results, self.collection.count() or 1)
        )
        items = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            try:
                data = json.loads(doc)
            except Exception:
                data = doc
            items.append({"key": meta.get("key"), "data": data, "meta": meta})
        return items

    def store_script(self, script: Dict):
        self.store("script:latest", script, {"type": "script"})

    def store_characters(self, characters: List[Dict]):
        for char in characters:
            name = char.get("name", "unknown").lower().replace(" ", "_")
            self.store(f"character:{name}", char, {"type": "character"})
        self.store("characters:all", characters, {"type": "character_list"})

    def get_all_characters(self) -> List[Dict]:
        result = self.retrieve("characters:all")
        return result if isinstance(result, list) else []

    def store_image_ref(self, char_name: str, image_path: str):
        self.store(
            f"image:{char_name.lower()}",
            {"character": char_name, "path": image_path},
            {"type": "image"}
        )


memory = VectorMemory()
