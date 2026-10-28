"""Tiny local knowledge base: keyword retrieval over markdown files in data/knowledge_base."""
import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
STOP = set(
    "the a an and or of to in on for with by is are was were be as that this it its from at into "
    "than then their they them can not only such more most also which".split()
)


def _stem(w: str) -> str:
    for suf in ("ic", "ing", "ed", "es", "s"):
        if len(w) > 5 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP]


@dataclass
class Doc:
    id: str
    title: str
    text: str

    @property
    def sentences(self) -> list:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", self.text) if s.strip()]


def load_docs() -> list:
    docs = []
    for p in sorted(KB_DIR.glob("*.md")):
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        text = " ".join(l.strip().lstrip("- ").strip() for l in lines[1:] if l.strip())
        docs.append(Doc(p.stem, title, text))
    return docs


def get_doc(doc_id: str):
    return next((d for d in load_docs() if d.id == doc_id), None)


def search(query: str, k: int = 3) -> list:
    q = set(tokenize(query))
    scored = []
    for d in load_docs():
        score = len(q & set(tokenize(d.text))) + 2 * len(q & set(tokenize(d.title)))
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]


def support(claim: str, doc_text: str) -> float:
    """Share of the claim's content words that appear in the cited document (0-1)."""
    c = set(tokenize(claim))
    return len(c & set(tokenize(doc_text))) / len(c) if c else 0.0
