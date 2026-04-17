from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import httpx
import os
import yaml
from pathlib import Path

app = FastAPI(title="LangChain Orchestrator")

HERMES_URL = os.getenv("HERMES_URL", "http://hermes:3000")
WIKI_DIR = Path("/app/hermes-data/wiki")

class QuestionRequest(BaseModel):
    question: str
    intent: Optional[str] = None
    format: Literal["legal", "academic", "technical", "executive", "data", "conversational", "medical", "financial", "troubleshooting"] = "academic"

class AnswerResponse(BaseModel):
    answer: str
    intent: str
    format: str
    confidence: float
    sources: list[str]
    metadata: dict

INTENT_KEYWORDS = {
    "legal": ["ley", "contrato", "demanda", "cláusula", "derecho", "fallo", "jurisprudencia", "artículo"],
    "academic": ["investigación", "estudio", "teoría", "metodología", "resultados", "abstract", "paper"],
    "technical": ["error", "código", "bug", "api", "endpoint", "deploy", "docker", "servidor"],
    "medical": ["paciente", "diagnóstico", "tratamiento", "clínico", "síntomas", "medicamento"],
    "financial": ["inversión", "rentabilidad", "balance", "ingresos", "egresos", "acciones", "banco"],
    "troubleshooting": ["problema", "error", "no funciona", "falla", "solucionar", "debug"],
    "executive": ["resumen", "ejecutivo", "overview", "objetivos", "metas", "kpi"],
    "data": ["datos", "análisis", "métricas", "gráfico", "estadística", "dashboard"],
    "conversational": ["hola", "qué tal", "contame", "cómo estás", "hablame"],
}

FORMAT_TEMPLATES = {
    "legal": {
        "structure": "VOTO | ANTECEDENTES | MARCO LEGAL | ANÁLISIS | FALLO",
        "fields": ["partes", "objeto", "jurisdiccion", "norma_aplicada", "considerandos", "sentencia"]
    },
    "academic": {
        "structure": "ABSTRACT | INTRODUCCIÓN | METODOLOGÍA | RESULTADOS | DISCUSIÓN | REFERENCIAS",
        "fields": ["tema", "hipotesis", "metodo", "conclusiones", "bibliografia"]
    },
    "technical": {
        "structure": "PROBLEMA | ERROR LOG | CAUSA RAÍZ | SOLUCIÓN | VERIFICACIÓN",
        "fields": ["error_type", "stack_trace", "root_cause", "fix", "test_steps"]
    },
}

def detect_intent(question: str) -> str:
    q = question.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in q)
    if max(scores.values()) == 0:
        return "conversational"
    return max(scores, key=scores.get)

def format_legal(query: str, context: str) -> dict:
    return {
        "tipo_documento": "Dictamen Legal",
        "materia": "General",
        "partes_involucradas": [],
        "normas_aplicadas": [],
        "analisis": context[:1000],
        "conclusion": f"Basado en la información consultada: {query}",
        "fuentes": []
    }

def format_academic(query: str, context: str) -> dict:
    return {
        "titulo": query[:100],
        "abstract": context[:500],
        "introduccion": context[:500],
        "metodologia": context[:300],
        "resultados": context[:500],
        "conclusiones": f"Análisis de: {query}",
        "referencias": []
    }

def format_technical(query: str, context: str) -> dict:
    return {
        "problema_descrito": query,
        "contexto_tecnico": context[:1000],
        "causa_probable": "Verificar logs del contenedor",
        "solucion_steps": ["1. Revisar healthcheck", "2. Ver logs", "3. Validar volumen"],
        "verificacion": "Probar endpoint /health"
    }

FORMATTERS = {
    "legal": format_legal,
    "academic": format_academic,
    "technical": format_technical,
}

@app.get("/health")
def health():
    return {"status": "ok", "service": "langchain-orchestrator"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    intent = req.intent or detect_intent(req.question)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            hermes_resp = await client.post(
                f"{HERMES_URL}/query",
                json={"question": req.question}
            )
            hermes_data = hermes_resp.json()
    except Exception as e:
        return AnswerResponse(
            answer=f"Error conectando con Hermes: {e}",
            intent=intent,
            format=req.format,
            confidence=0.0,
            sources=[],
            metadata={"error": str(e)}
        )

    formatter = FORMATTERS.get(req.format, lambda q, c: {"raw": c[:1000]})
    formatted = formatter(req.question, hermes_data.get("answer", ""))

    return AnswerResponse(
        answer=yaml.dump(formatted, allow_unicode=True, sort_keys=False),
        intent=intent,
        format=req.format,
        confidence=0.85,
        sources=hermes_data.get("sources", []),
        metadata={"chunks_used": len(hermes_data.get("sources", [])), "hermes_response": hermes_data}
    )

@app.post("/intents")
def detect_intents_endpoint(questions: list[str]):
    return {"intents": [detect_intent(q) for q in questions]}

@app.get("/documents")
def list_documents():
    if not WIKI_DIR.exists():
        return {"documents": [], "count": 0}
    files = sorted([f.name for f in WIKI_DIR.glob("*.md")])
    return {"documents": files, "count": len(files)}

@app.get("/formats")
def list_formats():
    return {
        "formats": list(FORMAT_TEMPLATES.keys()),
        "templates": FORMAT_TEMPLATES
    }
