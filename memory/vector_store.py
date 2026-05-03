"""
memory/vector_store.py
──────────────────────
Persistent memory layer using ChromaDB.
All agents read from and write to this shared store,
enabling continuity across workflow nodes.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_PATH, COLLECTION


class VectorMemory:
    """
    Wraps ChromaDB to give agents a simple key-value + semantic memory.

    Usage:
        memory.store("character:alice", {"name": "Alice", "age": 30})
        results = memory.query("female protagonist")
        memory.retrieve("character:alice")
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, key: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """Store data with a unique key. Returns document ID."""
        doc_id = str(uuid.uuid4())
        content = json.dumps(data) if not isinstance(data, str) else data
        meta = metadata or {}
        meta["key"] = key

        # Upsert: update if key exists
        existing = self.collection.get(where={"key": key})
        if existing["ids"]:
            self.collection.update(
                ids=existing["ids"],
                documents=[content],
                metadatas=[meta]
            )
            return existing["ids"][0]

        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta]
        )
        return doc_id

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve stored data by exact key."""
        results = self.collection.get(where={"key": key})
        if not results["documents"]:
            return None
        raw = results["documents"][0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def query(self, text: str, n_results: int = 5) -> List[Dict]:
        """Semantic similarity search over stored memory."""
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


# Singleton
memory = VectorMemory()
