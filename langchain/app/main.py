"""
Hermes LangChain API
Lee archivos Markdown del wiki generado por Hermes y los expone como REST API.
Expanded: upload, delete, folders, smart search.
"""
import os
import shutil
import uuid
import re
import unicodedata
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
import markdown
import asyncio

app = FastAPI(
    title="Hermes LangChain API",
    description="API que lee wikis Markdown generados por Hermes y los estructura con Pydantic",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WIKI_PATH = os.getenv("WIKI_PATH", "/hermes-data/wiki")
INPUT_PATH = os.getenv("INPUT_PATH", "/hermes-data/input")
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "/hermes-data/uploads")

# Ensure directories exist
Path(WIKI_PATH).mkdir(parents=True, exist_ok=True)
Path(INPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_PATH).mkdir(parents=True, exist_ok=True)


# --- Pydantic Models ---

class WikiPage(BaseModel):
    slug: str
    title: str
    content: str
    html_content: str
    links: list[str] = []
    backlinks: list[str] = []
    tags: list[str] = []
    source_file: str


class WikiIndex(BaseModel):
    total_pages: int
    pages: list[dict]
    tags: list[str]


class SearchResult(BaseModel):
    query: str
    folder: Optional[str] = None
    total: int
    results: list[dict]


class HealthResponse(BaseModel):
    status: str
    wiki_path: str
    wiki_exists: bool
    pages_count: int


class UploadResponse(BaseModel):
    status: str
    filename: str
    saved_to: str
    folder: Optional[str] = None


class FolderInfo(BaseModel):
    name: str
    path: str
    page_count: int
    pages: list[str]


class FolderListResponse(BaseModel):
    folders: list[str]
    default_wiki_pages: int


# --- Helpers ---

def extract_wiki_links(content: str) -> list[str]:
    """Extrae links estilo Obsidian [[enlace]]"""
    return re.findall(r'\[\[([^\]]+)\]\]', content)


def extract_tags(content: str) -> list[str]:
    """Extrae tags del frontmatter o del contenido"""
    tags = re.findall(r'^tags?\s*:\s*-\s*(.+)$', content, re.MULTILINE)
    flat = []
    for t in tags:
        flat.extend([x.strip() for x in t.split(',')])
    return [t for t in flat if t]


def extract_title(content: str, filename: str) -> str:
    """Extrae título del markdown o usa el filename"""
    h1 = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1:
        return h1.group(1).strip()
    return filename.replace('.md', '').replace('-', ' ').title()


def get_all_slugs(wiki_path: Path) -> dict[str, Path]:
    """Mapa slug -> filepath (solo raíz, no subcarpetas)"""
    if not wiki_path.exists():
        return {}
    return {f.stem: f for f in wiki_path.glob("*.md")}


def compute_backlinks(wiki_path: Path, slugs: dict[str, Path]) -> dict[str, list[str]]:
    """Calcula qué páginas linkean a cada página"""
    backlinks: dict[str, list[str]] = {s: [] for s in slugs}
    for slug, filepath in slugs.items():
        content = filepath.read_text(encoding='utf-8')
        linked = extract_wiki_links(content)
        for link in linked:
            link_slug = link.replace(' ', '-').lower()
            if link_slug in backlinks:
                backlinks[link_slug].append(slug)
    return backlinks


def markdown_to_html(md_content: str) -> str:
    """Convierte markdown a HTML"""
    return markdown.markdown(
        md_content,
        extensions=['fenced_code', 'tables', 'toc']
    )


def slugify(text: str) -> str:
    """Convierte texto a slug URL-safe"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text


def normalize(text: str) -> str:
    """Normaliza texto: minúsculas + elimina acentos para búsqueda sinítica"""
    text = text.lower().strip()
    # Eliminar acentos: á → a, é → e, etc.
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)  # quita diacríticos
    return text


def extract_article_by_number(content: str, query: str) -> str:
    """Extrae un artículo específico cuando la query contiene número de artículo"""
    # Buscar patrón "articulo 112" o "art. 112" o solo "112"
    match = re.search(r'(\d{3,})\b', query)
    if not match:
        return ""
    art_num = match.group(1)

    # Buscar el artículo en el documento (con o sin tilde, con o sin punto)
    # Patrones: **Artículo 112.** o **Art. 112** o Artículo 112.
    patterns = [
        rf'\*\*(?:Artículo|Art\.?)\s*{art_num}[^.]*\.?\*\*\s*(.*?)(?=\n\*\*[A-Z]|\n---|\n#|\Z)',
        rf'(?:^|\n)(?:Artículo|Art\.?)\s*{art_num}[^.]*\.?\s*\n(.*?)(?=\n(?:[A-ZÁÉÍÓÚ]{{2,}}|---)|\Z)',
    ]
    for pattern in patterns:
        m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            # Limpiar negritas
            text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
            # Limitar largo
            if len(text) > 600:
                text = text[:600] + '...'
            return text
    return ""


def extract_summary(content: str) -> str:
    """Extrae sección Resumen del documento para respuestas inteligentes"""
    # Buscar patrón ## Resumen o ## Resumen/Ejecutivo
    patterns = [
        r'(?:^|\n)(?:#+\s*RESUMEN|Resumen\s*:?|RESUMEN\s*EJECUTIVO|Sumilla)(?:\s*[–:-]?\s*)(.*?)(?=\n##|\n#|$)',
        r'(?:^|\n)(?:#+\s*ABSTRACT)(.*?)(?=\n##|\n#|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            # Limpiar caracteres especiales
            text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)  # bold/italic
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # links
            return text[:500]
    return ""


def smart_search_content(wiki_path: Path, query: str, folder: Optional[str] = None) -> list[dict]:
    """
    Búsqueda inteligente con normalización de acentos y extracción de artículos.
    """
    q_normalized = normalize(query)
    results = []

    # Detectar si es consulta general tipo "de qué trata"
    general_patterns = ['de qué trata', 'de que trata', 'resumen del documento',
                        'resumen general', 'de que se trata', 'overview', 'summary',
                        'de que trata el', 'de qué se trata', 'que trata']
    is_general_query = any(p in q_normalized for p in general_patterns)

    # Detectar si es consulta de artículo específico ("articulo 112" o "112")
    has_article_number = bool(re.search(r'\d{3,}', query))

    # Determinar path de búsqueda
    search_path = Path(wiki_path)
    if folder and folder != "root" and folder != "wiki":
        search_path = wiki_path / folder
        if not search_path.exists():
            search_path = wiki_path

    for f in search_path.glob("*.md"):
        if f.is_dir():
            continue
        content = f.read_text(encoding='utf-8')
        content_normalized = normalize(content)

        # Búsqueda con normalización de acentos
        if q_normalized not in content_normalized:
            continue

        title = extract_title(content, f.stem)

        if is_general_query:
            # Para consultas generales, priorizar sección Resumen
            summary = extract_summary(content)
            preview = summary if summary else content[:200]
        elif has_article_number:
            # Extraer artículo específico
            article_text = extract_article_by_number(content, query)
            if article_text:
                preview = f"**Artículo encontrado:**\n\n{article_text}"
            else:
                # fallback: snippet normal
                idx = content_normalized.index(q_normalized)
                preview = content[max(0, idx-50):idx+150].strip()
        else:
            # Búsqueda normal por snippets con normalización
            idx = content_normalized.index(q_normalized)
            preview = content[max(0, idx-50):idx+150].strip()

        # Calcular folder relativo
        rel_folder = "root"
        if f.parent != wiki_path:
            rel_folder = f.parent.name

        results.append({
            "slug": f.stem,
            "title": title,
            "preview": preview,
            "file": f.name,
            "folder": rel_folder
        })

    return results


# --- Routes ---

@app.get("/health", response_model=HealthResponse)
def health():
    wiki_path = Path(WIKI_PATH)
    pages = list(wiki_path.glob("*.md")) if wiki_path.exists() else []
    return HealthResponse(
        status="ok",
        wiki_path=WIKI_PATH,
        wiki_exists=wiki_path.exists(),
        pages_count=len(pages)
    )


@app.get("/")
def root():
    return {
        "service": "Hermes LangChain API",
        "version": "1.1.0",
        "endpoints": [
            "/health",
            "/wiki/index",
            "/wiki/page/{slug}",
            "/wiki/random",
            "/search?q=query&folder=root",
            "/upload",
            "/wiki/page/{slug} (DELETE)",
            "/folders/list",
            "/folders/create",
            "/folders/{folder}",
            "/folders/{folder}/pages",
            "/input/list"
        ]
    }


@app.get("/wiki/index", response_model=WikiIndex)
def wiki_index(folder: Optional[str] = None):
    wiki_path = Path(WIKI_PATH)
    if not wiki_path.exists():
        raise HTTPException(status_code=404, detail="Wiki path not found")

    search_path = wiki_path
    if folder and folder != "root" and folder != "wiki":
        search_path = wiki_path / folder
        if not search_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder '{folder}' not found")

    pages = []
    tags_all = set()

    for f in search_path.glob("*.md"):
        if f.is_dir():
            continue
        content = f.read_text(encoding='utf-8')
        slug = f.stem
        title = extract_title(content, slug)
        tags = extract_tags(content)
        tags_all.update(tags)
        pages.append({
            "slug": slug,
            "title": title,
            "tags": tags,
            "file": f.name,
            "folder": folder if folder and folder != "root" else "root"
        })

    return WikiIndex(
        total_pages=len(pages),
        pages=sorted(pages, key=lambda x: x['title']),
        tags=sorted(tags_all)
    )


@app.get("/wiki/page/{slug}", response_model=WikiPage)
def get_page(slug: str, folder: Optional[str] = None):
    wiki_path = Path(WIKI_PATH)
    slugs = get_all_slugs(wiki_path)

    if slug not in slugs:
        raise HTTPException(status_code=404, detail=f"Page '{slug}' not found")

    filepath = slugs[slug]
    content = filepath.read_text(encoding='utf-8')
    backlinks_map = compute_backlinks(wiki_path, slugs)

    return WikiPage(
        slug=slug,
        title=extract_title(content, slug),
        content=content,
        html_content=markdown_to_html(content),
        links=extract_wiki_links(content),
        backlinks=backlinks_map.get(slug, []),
        tags=extract_tags(content),
        source_file=filepath.name
    )


@app.delete("/wiki/page/{slug}")
def delete_page(slug: str, folder: Optional[str] = None):
    """Elimina una página del wiki"""
    wiki_path = Path(WIKI_PATH)

    if folder and folder != "root":
        filepath = wiki_path / folder / f"{slug}.md"
    else:
        filepath = wiki_path / f"{slug}.md"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Page '{slug}' not found")

    filepath.unlink()
    return {"status": "deleted", "slug": slug, "file": str(filepath)}


@app.get("/wiki/random", response_model=WikiPage)
def random_page(folder: Optional[str] = None):
    import random
    wiki_path = Path(WIKI_PATH)

    search_path = wiki_path
    if folder and folder != "root":
        search_path = wiki_path / folder

    pages = list(search_path.glob("*.md")) if search_path.exists() else []
    if not pages:
        raise HTTPException(status_code=404, detail="No pages found")
    random_file = random.choice(pages)
    slug = random_file.stem
    return get_page(slug)


@app.get("/search", response_model=SearchResult)
def search(q: str = "", folder: Optional[str] = None):
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short (min 2 chars)")

    wiki_path = Path(WIKI_PATH)
    results = smart_search_content(wiki_path, q, folder)

    return SearchResult(
        query=q,
        folder=folder,
        total=len(results),
        results=results
    )


# === UPLOAD ===

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    folder: Optional[str] = Form(None),
    process: bool = Form(True)  # si True, intenta extraer texto y crear .md
):
    """
    Sube un archivo PDF al wiki.
    Si process=True, extrae el texto del PDF y genera una entrada en /wiki.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validar tipo
    allowed_types = ['.pdf', '.md', '.txt', '.doc', '.docx']
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo no soportado: {ext}. Solo: {', '.join(allowed_types)}"
        )

    # Guardar en input/
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    input_filepath = Path(INPUT_PATH) / safe_filename

    content = await file.read()
    with open(input_filepath, "wb") as f:
        f.write(content)

    result_folder = folder if folder else "root"

    # Si es PDF y process=True, extraer texto y crear .md en wiki
    if ext == '.pdf' and process:
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, extract_pdf_text, input_filepath
            )
            if text and text.strip():
                md_filename = slugify(Path(file.filename).stem) + ".md"
                wiki_dest = Path(WIKI_PATH) / md_filename

                # Generar markdown
                md_content = generate_md_from_pdf(
                    file.filename,
                    text,
                    Path(file.filename).stem
                )

                with open(wiki_dest, "w", encoding="utf-8") as f:
                    f.write(md_content)

                return UploadResponse(
                    status="indexed",
                    filename=file.filename,
                    saved_to=str(input_filepath),
                    folder=result_folder
                )
            else:
                return UploadResponse(
                    status="uploaded_no_text",
                    filename=file.filename,
                    saved_to=str(input_filepath),
                    folder=result_folder
                )
        except Exception as e:
            return UploadResponse(
                status="uploaded_error",
                filename=file.filename,
                saved_to=str(input_filepath),
                folder=result_folder
            )

    return UploadResponse(
        status="uploaded",
        filename=file.filename,
        saved_to=str(input_filepath),
        folder=result_folder
    )


def extract_pdf_text(pdf_path: Path) -> str:
    """Extrae texto de PDF usando pypdf (sincrónico)"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text() + "\n"
            except Exception:
                text += "\n"
        return text
    except ImportError:
        return "[pypdf no instalado - no se pudo extraer texto]"
    except Exception as e:
        return f"[Error extrayendo PDF: {e}]"


def generate_md_from_pdf(original_name: str, text: str, title: str) -> str:
    """Genera entrada markdown desde texto extraído de PDF"""
    # Extraer primeras líneas como resumen
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    first_lines = ' '.join(lines[:20])[:300]

    # Detectar tipo de documento
    doc_type = "documento"
    if any(k in text[:500].upper() for k in ['REGLAMENTO', 'RESOLUCIÓN', 'CONTRATO', 'CONVOCATORIA']):
        doc_type = "normativo"
    elif any(k in text[:500].upper() for k in ['CONVOCATORIA', 'LLAMADO', 'BASES']):
        doc_type = "convocatoria"

    # Buscar artículos
    articles = re.findall(r'(?:ARTÍCULO|Artículo|Art\.)\s*(\d+|[IVXLCDM]+)[\.\):\s]*(.*?)(?=\n(?:ARTÍCULO|Artículo|Art\.|$))',
                          text[:5000], re.DOTALL | re.IGNORECASE)
    article_preview = ""
    if articles:
        sample = articles[:3]
        article_preview = "Abarca artículos: " + ", ".join(
            f"{num}" for num, _ in sample
        )

    fm = f"""---
slug: {slugify(title)}
title: {title}
type: {doc_type}
source: {original_name}
date: 2026-04-17
---

# {title}

## Resumen

{first_lines}

{article_preview}

## Contenido

(text extracted from PDF - {len(text)} characters)

```
{text[:2000]}...
```

"""
    return fm


# === FOLDERS ===

@app.get("/folders/list", response_model=FolderListResponse)
def list_folders():
    """Lista todas las carpetas en el wiki (incluyendo raíz)"""
    wiki_path = Path(WIKI_PATH)
    folders = []

    if wiki_path.exists():
        for item in wiki_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                page_count = len(list(item.glob("*.md")))
                folders.append(item.name)

    # Añadir raíz
    root_pages = len(list(wiki_path.glob("*.md"))) if wiki_path.exists() else 0

    return FolderListResponse(
        folders=sorted(folders),
        default_wiki_pages=root_pages
    )


@app.post("/folders/create")
def create_folder(name: str = Form(...)):
    """Crea una nueva carpeta en el wiki"""
    if not name or len(name) < 1:
        raise HTTPException(status_code=400, detail="Folder name required")

    # Sanitizar nombre
    safe_name = slugify(name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    folder_path = Path(WIKI_PATH) / safe_name
    if folder_path.exists():
        raise HTTPException(status_code=409, detail=f"Folder '{safe_name}' already exists")

    folder_path.mkdir(parents=True, exist_ok=True)

    return {"status": "created", "folder": safe_name, "path": str(folder_path)}


@app.delete("/folders/{folder}")
def delete_folder(folder: str):
    """Elimina una carpeta y todo su contenido"""
    if folder in ('root', 'wiki', '.', '..'):
        raise HTTPException(status_code=400, detail="Cannot delete protected folder")

    folder_path = Path(WIKI_PATH) / folder
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Folder '{folder}' not found")

    # Eliminar contenido
    shutil.rmtree(folder_path)

    return {"status": "deleted", "folder": folder}


@app.get("/folders/{folder}/pages")
def get_folder_pages(folder: str):
    """Lista todas las páginas en una carpeta"""
    if folder in ('root', 'wiki'):
        # Redirigir al índice general
        return wiki_index(folder=None)

    folder_path = Path(WIKI_PATH) / folder
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Folder '{folder}' not found")

    pages = []
    for f in folder_path.glob("*.md"):
        content = f.read_text(encoding='utf-8')
        pages.append({
            "slug": f.stem,
            "title": extract_title(content, f.stem),
            "file": f.name
        })

    return {"folder": folder, "pages": sorted(pages, key=lambda x: x['title']), "count": len(pages)}


# === INPUT FILES ===

@app.get("/input/list")
def list_input_files():
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        return {"files": [], "path": INPUT_PATH}

    files = [{"name": f.name, "size": f.stat().st_size, "type": f.suffix}
             for f in sorted(input_path.iterdir())]
    return {"path": INPUT_PATH, "files": files}


@app.delete("/input/{filename}")
def delete_input_file(filename: str):
    """Elimina un archivo de la carpeta input"""
    filepath = Path(INPUT_PATH) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    filepath.unlink()
    return {"status": "deleted", "filename": filename}


@app.get("/input/{filename}")
def read_input_file(filename: str):
    filepath = Path(INPUT_PATH) / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
