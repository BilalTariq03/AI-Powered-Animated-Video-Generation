"""
shared/utils/vector_store.py
─────────────────────────────
Persistent memory layer using ChromaDB.
Falls back to a no-op stub if chromadb is not installed so that
Phase 2 / Phase 3 can run without the full dependency set.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from config import CHROMA_PATH, COLLECTION

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class _StubCollection:
    """No-op ChromaDB collection used when chromadb is not installed."""
    def get(self, **_):       return {"ids": [], "documents": []}
    def add(self, **_):       pass
    def update(self, **_):    pass
    def query(self, **_):     return {"documents": [[]], "metadatas": [[]]}
    def count(self):          return 0


class VectorMemory:
    def __init__(self):
        if _CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
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
