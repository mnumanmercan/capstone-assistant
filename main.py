import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core.ingest import load_document, chunk_text, embed_chunks
from core.retrieve import retrieve
from core.rag import build_context, generate, stream_generate, RagAnswer
from core.agent import agent

DATA_PATH = Path(__file__).parent / "data" / "life-changing-daily-habit.txt"
CACHE_PATH = Path(__file__).parent / "data" / ".index_cache.json"

# Cosine benzerlik eşikleri. Kendi verinde kalibre edilmeli (aşağıdaki nota bak).
HIGH_THRESHOLD = 0.70
NEUTRAL_THRESHOLD = 0.55


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Modeller
# ----------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AgentResponse(BaseModel):
    answer: str


@dataclass
class Retrieval:
    """Bir sorunun retrieval sonucu: chunk'lar, skorlar ve hazır context."""

    question: str
    chunks: list[str]
    scores: list[float]
    context: str

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def reliability(self) -> Literal["high", "neutral", "low"]:
        """Modele SORMADAN, retrieval skorundan hesaplanır: bedava + deterministik."""
        if self.top_score >= HIGH_THRESHOLD:
            return "high"
        if self.top_score >= NEUTRAL_THRESHOLD:
            return "neutral"
        return "low"

    def as_sources(self) -> list[dict]:
        return [
            {"id": i, "score": round(s, 4), "preview": c[:120]}
            for i, (c, s) in enumerate(zip(self.chunks, self.scores))
        ]


# ----------------------------------------------------------------------------
# Dependency: /ask ve /ask/stream'in ortak retrieval adımı
# ----------------------------------------------------------------------------
async def get_retrieval(payload: AskRequest, request: Request) -> Retrieval:
    """React custom hook gibi: endpoint 'bana retrieval lazım' der, gerisi buranın işi."""
    # retrieve blocking -> async fonksiyonun içinde threadpool'a atılmalı
    ilgili_chunklar, scores = await run_in_threadpool(
        retrieve,
        payload.question,
        request.app.state.chunks,
        request.app.state.doc_vecs,
        payload.k,
    )
    return Retrieval(
        question=payload.question,
        chunks=ilgili_chunklar,
        scores=scores,
        context=build_context(ilgili_chunklar),
    )


def sse(event: str, data) -> str:
    """Python değerini SSE mesajına çevirir: 'event: <tip>\\ndata: <json>\\n\\n'."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------------
# Endpoint'ler
# ----------------------------------------------------------------------------
@app.get("/health")
async def get_status(request: Request):
    return {"status": "ok", "chunks": len(request.app.state.chunks)}


@app.post("/ask", response_model=RagAnswer)
async def ask(rtr: Retrieval = Depends(get_retrieval)) -> RagAnswer:
    try:
        answer = await run_in_threadpool(generate, rtr.question, rtr.context)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"LLM servisi yanıt vermedi: {e}")

    # reliability'yi modelin tahminiyle değil, ölçülen skorla değiştiriyoruz.
    answer.reliability = rtr.reliability
    return answer


@app.post("/ask/stream")
async def ask_stream(rtr: Retrieval = Depends(get_retrieval)) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        # 1) Kaynaklar + güven: modelden beklemeye gerek yok, ilk milisaniyede hazır.
        yield sse("sources", rtr.as_sources())
        yield sse("reliability", rtr.reliability)

        # 2) Token'lar
        try:
            async for text in stream_generate(rtr.question, rtr.context):
                yield sse("token", text)
        except Exception as e:
            # Stream başladıktan sonra status kodu değiştirilemez -> hata bir event'tir.
            yield sse("error", str(e))
            return

        # 3) Bitiş sinyali
        yield sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent", response_model=AgentResponse)
def run_agent(payload: AgentRequest, request: Request) -> AgentResponse:
    """Agentic RAG: aramayı biz değil, model yapar.

    - Fonksiyon adı 'agent' DEĞİL: import ettiğimiz agent()'ı ezerdi.
    - async DEĞİL: agent() içinde senkron messages.create var, hem de döngüde.
      FastAPI bu fonksiyonu threadpool'a atar.
    - Depends(get_retrieval) YOK: agent'a hazır context değil, INDEX veriyoruz.
      Ne zaman ve hangi sorguyla arayacağına kendi karar verecek.
    """
    try:
        answer = agent(
            payload.question,
            request.app.state.chunks,
            request.app.state.doc_vecs,
        )
    except Exception as e:
        # Agent döngüsü tipli exception fırlatmıyor (Faz 6'da düzelteceğiz),
        # o yüzden geniş yakalayıp upstream hatası olarak raporluyoruz.
        raise HTTPException(status_code=502, detail=f"Agent çalışırken hata: {e}")

    return AgentResponse(answer=answer)
