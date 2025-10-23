#!/usr/bin/env python3
import pathlib, json, hashlib
from typing import Iterable

root = pathlib.Path(__file__).resolve().parents[1]
pdf_root = root/"docs"/"pdfs"
pdf_root.mkdir(parents=True, exist_ok=True)
index = root/"data"/"rag_index.jsonl"
index.parent.mkdir(parents=True, exist_ok=True)

def walk_sources() -> Iterable[pathlib.Path]:
    for repo in (root/"repos").glob("*"):
        for p in repo.rglob("*"):
            if p.suffix.lower() in [".pdf",".md",".txt",".rtf"]:
                yield p

def extract_pdf(path: pathlib.Path) -> str:
    try:
        import pypdf
        r = pypdf.PdfReader(str(path))
        return "\n".join([(p.extract_text() or "") for p in r.pages])
    except Exception:
        return ""

def chunk(text, n=1200):
    for i in range(0, len(text), n):
        yield text[i:i+n]

with index.open("w", encoding="utf-8") as out:
    for src in walk_sources():
        try:
            if src.suffix.lower() == ".pdf":
                text = extract_pdf(src)
            else:
                text = src.read_text(encoding="utf-8", errors="ignore")
            for i, ch in enumerate(chunk(text)):
                rid = hashlib.sha256(f"{src}:{i}".encode()).hexdigest()[:16]
                out.write(json.dumps({
                    "id": rid, "source": str(src.relative_to(root)), "part": i, "text": ch
                }, ensure_ascii=False) + "\n")
        except Exception:
            continue
print("RAG index ->", index)