import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Doc:
    id: str
    title: str
    text: str
    source_id: Optional[str] = None
    chunk_id: Optional[int] = None

class SimpleRAG:
    def __init__(
        self,
        corpus_path: str,
        embed_model: str = "all-MiniLM-L6-v2",
        chunk_chars: int = 800,
        chunk_overlap: int = 80,
        cache_dir: Optional[str] = "data/rag_cache",
    ):
        self.embedder = SentenceTransformer(embed_model)
        self.docs: List[Doc] = []
        self.index = None
        self.embeddings = None
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir and self._load_cache():
            return

        self._load_corpus(corpus_path)
        self._build_index()
        if self.cache_dir:
            self._save_cache()

    def _load_corpus(self, corpus_path: str):
        p = Path(corpus_path)
        with p.open() as f:
            for line in f:
                d = json.loads(line)
                self.docs.extend(self._chunk_doc(d))

    def _chunk_doc(self, d: dict) -> List[Doc]:
        text = d["text"]
        title = d["title"]
        source_id = d["id"]
        if self.chunk_chars <= 0 or len(text) <= self.chunk_chars:
            return [Doc(source_id, title, text, source_id=source_id, chunk_id=0)]

        chunks = []
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_chars)
            window = text[start:end]
            if end < len(text):
                split_at = max(window.rfind("\n"), window.rfind(" "))
                if split_at > self.chunk_chars - 120:
                    end = start + split_at
                    window = text[start:end]
            window = window.strip()
            if window:
                doc_id = f"{source_id}#chunk{chunk_id}"
                chunks.append(Doc(doc_id, title, window, source_id=source_id, chunk_id=chunk_id))
                chunk_id += 1
            if end == len(text):
                break
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end
        return chunks

    def _build_index(self):
        texts = [f"{d.title}\n{d.text}" for d in self.docs]
        emb = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb.astype(np.float32)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized IP
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Doc, float]]:
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append((self.docs[int(idx)], float(score)))
        return results

    def _cache_paths(self):
        if not self.cache_dir:
            return None
        return {
            "docs": self.cache_dir / "docs.jsonl",
            "embeddings": self.cache_dir / "embeddings.npy",
            "index": self.cache_dir / "index.faiss",
        }

    def _load_cache(self) -> bool:
        paths = self._cache_paths()
        if not paths:
            return False
        if not (paths["docs"].exists() and paths["embeddings"].exists() and paths["index"].exists()):
            return False

        self.docs = []
        with paths["docs"].open() as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(
                    Doc(
                        d["id"],
                        d["title"],
                        d["text"],
                        source_id=d.get("source_id"),
                        chunk_id=d.get("chunk_id"),
                    )
                )
        self.embeddings = np.load(paths["embeddings"]).astype(np.float32)
        self.index = faiss.read_index(str(paths["index"]))
        return True

    def _save_cache(self):
        paths = self._cache_paths()
        if not paths:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with paths["docs"].open("w") as f:
            for d in self.docs:
                f.write(
                    json.dumps(
                        {
                            "id": d.id,
                            "title": d.title,
                            "text": d.text,
                            "source_id": d.source_id,
                            "chunk_id": d.chunk_id,
                        }
                    )
                    + "\n"
                )
        np.save(paths["embeddings"], self.embeddings)
        faiss.write_index(self.index, str(paths["index"]))

def format_context(results: List[Tuple[Doc, float]]) -> str:
    blocks = []
    for d, score in results:
        source = d.source_id or d.id
        chunk = f" chunk={d.chunk_id}" if d.chunk_id is not None else ""
        blocks.append(
            f"[source:{source}{chunk} score={score:.3f} title={d.title}]\n{d.text}".strip()
        )
    return "\n\n".join(blocks)
