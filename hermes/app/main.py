from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
import uuid
import re
from pathlib import Path

app = FastAPI(title="Hermes Wiki Generator")

DATA_DIR = Path("/app/hermes-data")
WIKI_DIR = DATA_DIR / "wiki"
UPLOAD_DIR = DATA_DIR / "uploads"
WIKI_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"[ERROR extracting PDF: {e}]"

def chunk_text(text: str, chunk_size: int = 800) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

def generate_embedding(text: str) -> list[float]:
    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text[:2048]},
            timeout=30.0
        )
        return response.json().get("embedding", [])
    except Exception:
        return [0.0] * 768

def yaml_frontmatter(title: str, tags: list[str], chunk_index: int, total: int) -> str:
    date = "2026-04-17"
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    return f"---\ntitle: {title}\ndate: {date}\ntags:\n{tags_yaml}\nchunk: {chunk_index + 1}/{total}\n---\n\n"

def markdown_to_wiki_links(text: str) -> str:
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}\b', text)
    entities = list(dict.fromkeys(words))[:10]
    for entity in entities:
        safe = entity.replace(" ", "")
        text = re.sub(rf'\b{re.escape(entity)}\b', f'[[{safe}]]', text)
    return text

def add_tags(text: str, keywords: list[str]) -> str:
    tags = [kw.lower().replace(" ", "-") for kw in keywords[:5]]
    return text + "\n\n" + " ".join(f"#{tag}" for tag in tags)

@app.get("/health")
def health():
    return {"status": "ok", "service": "hermes-wiki"}

@app.post("/index")
async def index_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo PDFs")

    file_id = str(uuid.uuid4())[:8]
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return JSONResponse({"error": "No se pudo extraer texto del PDF"}, status_code=422)

    chunks = chunk_text(text)
    title_base = Path(file.filename).stem.replace(" ", "-")[:50]

    created_files = []
    for i, chunk in enumerate(chunks):
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', chunk[:200])
        tags = list(dict.fromkeys(keywords))[:8] if keywords else ["document"]

        content = yaml_frontmatter(f"{title_base}-{i+1}", tags, i, len(chunks))
        content += f"## Chunk {i+1}/{len(chunks)}\n\n{chunk}\n\n"
        content = markdown_to_wiki_links(content)
        content = add_tags(content, tags)

        chunk_file = WIKI_DIR / f"{title_base}-{i+1}.md"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(content)
        created_files.append(str(chunk_file.name))

        emb = generate_embedding(chunk)
        print(f"[Hermes] Chunk {i+1}/{len(chunks)} embedded, dim={len(emb)}")

    return {
        "status": "indexed",
        "filename": file.filename,
        "chunks": len(chunks),
        "files": created_files,
        "wiki_dir": str(WIKI_DIR)
    }

@app.get("/documents")
def list_documents():
    files = sorted([f.name for f in WIKI_DIR.glob("*.md")])
    return {"documents": files, "count": len(files)}

@app.get("/wiki/{filename}")
def get_wiki(filename: str):
    file_path = WIKI_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No encontrado")
    return {"content": open(file_path).read(), "file": filename}

@app.post("/search")
def search_documents(query: str = ""):
    if not query.strip():
        return {"results": [], "query": query}

    query_lower = query.lower()
    results = []
    for md_file in WIKI_DIR.glob("*.md"):
        content = open(md_file).read().lower()
        if query_lower in content:
            lines = content.split("\n")
            snippets = [l.strip() for l in lines if query_lower in l.strip()][:3]
            results.append({
                "file": md_file.name,
                "matched": len(snippets),
                "snippets": snippets
            })
    return {"query": query, "results": results[:10]}

@app.post("/query")
def query_wiki(question: str = ""):
    search_resp = search_documents(question)
    if not search_resp["results"]:
        return {"answer": "No encontré información relevante.", "sources": []}

    context_chunks = []
    for r in search_resp["results"][:3]:
        try:
            content = open(WIKI_DIR / r["file"]).read()
            context_chunks.append(content[:500])
        except:
            pass

    context = "\n---\n".join(context_chunks)
    return {
        "answer": f"[Contexto recuperado: {len(context_chunks)} chunks]\n\n{context[:1000]}...",
        "sources": [r["file"] for r in search_resp["results"]]
    }
