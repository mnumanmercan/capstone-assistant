from typing import AsyncIterator, Literal

from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from config import settings

client = Anthropic(api_key=settings.anthropic_api_key)
async_client = AsyncAnthropic(api_key=settings.anthropic_api_key)


class RagAnswer(BaseModel):
    answer: str
    sources: list[int]
    reliability: Literal["high", "neutral", "low"]


# --- Ortak grounding kuralları: iki modda da aynı ---
GROUNDING_RULES = (
    "Sen yalnızca sağlanan kaynaklara dayanarak cevap veren bir asistansın.\n\n"
    "Kurallar:\n"
    "1. SADECE <kaynak> tag'lerinin içindeki bilgiyi kullan. Dışarıdan bilgi ekleme.\n"
    "2. Cevap kaynaklarda yoksa aynen şunu yaz: 'Bu bilgi kaynaklarda bulunmuyor.' "
    "Asla uydurma.\n"
)

# Structured mod: modelin dolduracağı alanların adlarını DOĞRU yazıyoruz.
STRUCTURED_SYSTEM = GROUNDING_RULES + (
    "3. 'sources' alanına gerçekten kullandığın kaynak id'lerini yaz.\n"
    "4. 'reliability' alanını, kaynakların soruyu ne kadar doğrudan yanıtladığına "
    "göre belirle.\n"
)

# Stream modu: JSON yok, düz metin akışı.
STREAM_SYSTEM = GROUNDING_RULES + (
    "3. Düz metin olarak, akıcı bir şekilde cevap ver. JSON veya etiket üretme.\n"
)


def build_context(chunks: list[str]) -> str:
    return "\n".join(
        f'<kaynak id="{i}">{chunk}</kaynak>'
        for i, chunk in enumerate(chunks)
    )


def _user_message(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question: str, context: str) -> RagAnswer:
    """Senkron + structured output. /ask endpoint'i bunu kullanır."""
    try:
        response = client.messages.parse(
            model="claude-haiku-4-5",
            max_tokens=settings.max_tokens,
            system=STRUCTURED_SYSTEM,
            messages=[{"role": "user", "content": _user_message(question, context)}],
            output_format=RagAnswer,
        )
    except Exception as e:
        raise RuntimeError(f"LLM call is unsuccess: {e}")

    return response.parsed_output


async def stream_generate(question: str, context: str) -> AsyncIterator[str]:
    """Async + token akışı. /ask/stream endpoint'i bunu kullanır.

    Normal bir fonksiyon gibi 'return' etmez; ürettikçe 'yield' eder.
    Çağıran taraf 'async for' ile tüketir.
    """
    async with async_client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=settings.max_tokens,
        system=STREAM_SYSTEM,
        messages=[{"role": "user", "content": _user_message(question, context)}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


if __name__ == "__main__":
    from pathlib import Path
    from core.ingest import load_document, chunk_text, embed_chunks
    from core.retrieve import retrieve

    yol = Path(__file__).parent.parent / "data" / "life-changing-daily-habit.txt"
    chunks = chunk_text(load_document(yol))
    doc_vecs = embed_chunks(chunks)

    soru = "Why does proving your inner voice wrong help you?"
    ilgili_chunklar, skorlar = retrieve(soru, chunks, doc_vecs, k=3)
    context = build_context(ilgili_chunklar)

    cevap = generate(soru, context)
    print("Cevap:", cevap.answer)
    print("Kullanılan kaynaklar:", cevap.sources)
    print("Güven:", cevap.reliability)
