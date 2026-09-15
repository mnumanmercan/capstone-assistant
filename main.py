import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core.ingest import load_document, chunk_text, embed_chunks
from core.retrieve import retrieve
from core.rag import build_context, generate, stream_generate, RagAnswer

DATA_PATH = Path(__file__).parent / "data" / "life-changing-daily-habit.txt"
CACHE_PATH = Path(__file__).parent / "data" / ".index_cache.json"


def build_index(path: Path) -> tuple[list[str], list[list[float]]]:
    """Dokümanı chunk'la ve embed et. İçerik değişmediyse diskteki cache'i kullan."""
    text = load_document(path)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if cache.get("hash") == text_hash:
            print(f"✅ Index cache'ten yüklendi ({len(cache['chunks'])} chunk) — API çağrısı yok")
            return cache["chunks"], cache["vecs"]

    print("⏳ Index kuruluyor (Voyage API çağrısı yapılıyor)...")
    chunks = chunk_text(text)
    vecs = embed_chunks(chunks)

    CACHE_PATH.write_text(
        json.dumps({"hash": text_hash, "chunks": chunks, "vecs": vecs}),
        encoding="utf-8",
    )
    print(f"✅ Index kuruldu ve cache'lendi ({len(chunks)} chunk)")
    return chunks, vecs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: henüz port dinlenmiyor, blocking kod burada SORUN DEĞİL ---
    app.state.chunks, app.state.doc_vecs = build_index(DATA_PATH)
    yield
    # --- SHUTDOWN ---
    print("👋 Kapanıyor, index bellekten bırakılıyor.")
    app.state.chunks = None
    app.state.doc_vecs = None


app = FastAPI(title="Capstone Assistant", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


def sse(event: str, data) -> str:
    """Bir Python değerini SSE mesaj formatına çevirir.

    Format: 'event: <tip>\\ndata: <json>\\n\\n'  (çift newline = mesaj sonu)
    ensure_ascii=False -> Türkçe karakterler bozulmasın.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def get_status(request: Request):
    return {"status": "ok", "chunks": len(request.app.state.chunks)}


@app.post("/ask", response_model=RagAnswer)
def ask(payload: AskRequest, request: Request) -> RagAnswer:
    # async DEĞİL: retrieve + generate blocking. FastAPI threadpool'a atar.
    chunks = request.app.state.chunks
    doc_vecs = request.app.state.doc_vecs

    ilgili_chunklar = retrieve(payload.question, chunks, doc_vecs, k=payload.k)
    context = build_context(ilgili_chunklar)

    try:
        return generate(payload.question, context)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"LLM servisi yanıt vermedi: {e}")


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    # async DOĞRU: stream_generate await'li bir async generator.
    chunks = request.app.state.chunks
    doc_vecs = request.app.state.doc_vecs

    # retrieve HÂLÂ blocking -> event loop'u kilitlememek için threadpool'a at.
    ilgili_chunklar = await run_in_threadpool(
        retrieve, payload.question, chunks, doc_vecs, payload.k
    )
    context = build_context(ilgili_chunklar)

    async def event_stream() -> AsyncIterator[str]:
        # 1) Kaynaklar: modelden beklemeye gerek yok, elimizde hazır.
        yield sse("sources", [
            {"id": i, "preview": c[:120]}
            for i, c in enumerate(ilgili_chunklar)
        ])

        # 2) Token'lar: model ürettikçe geçir.
        try:
            async for text in stream_generate(payload.question, context):
                yield sse("token", text)
        except Exception as e:
            # Stream BAŞLADIKTAN sonra HTTP status kodu değiştirilemez (header gitti).
            # Hatayı bu yüzden bir event olarak göndeririz.
            yield sse("error", str(e))
            return

        # 3) Bitti sinyali: client "done" görmeden akışı kapatmamalı.
        yield sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
