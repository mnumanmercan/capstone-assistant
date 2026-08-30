from pathlib import Path
import voyageai
from config import settings
from pydantic import BaseModel
from typing import Literal
from anthropic import Anthropic

client = Anthropic(api_key=settings.anthropic_api_key)
vo = voyageai.Client(api_key=settings.voyage_api_key)

class RagAnswer(BaseModel):
    answer: str
    sources: list[int]
    reliability: Literal["high", "neutral", "low"]

def build_context(chunks: list[str]) -> str:
    return "\n".join(
        f'<kaynak id="{i}">{chunk}</kaynak>'
        for i, chunk in enumerate(chunks)
    )

def generate(question: str, context: str) -> RagAnswer:
    system_prompt = (
        "Sen yalnızca sağlanan kaynaklara dayanarak cevap veren bir asistansın.\n\n"
        "Kurallar:\n"
        "1. SADECE <kaynak> tag'lerinin içindeki bilgiyi kullan. Dışarıdan bilgi "
        "ekleme.\n"
        "2. Cevap kaynaklarda yoksa aynen şunu yaz: 'Bu bilgi kaynaklarda "
        "bulunmuyor.' Asla uydurma.\n"
        "3. 'kullanilan_kaynaklar' alanına gerçekten kullandığın kaynak id'lerini "
        "yaz.\n"
        "4. 'guven' alanını, kaynakların soruyu ne kadar doğrudan yanıtladığına "
        "göre belirle."
    )

    user_message = f"Context: \n{context}\n\n Question: {question}"

    try:
        response = client.messages.parse(
            model="claude-haiku-4-5",
            max_tokens=settings.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            output_format=RagAnswer
        )
    except Exception as e:
        raise RuntimeError(f"LLM call is unsuccess: {e}")

    return response.parsed_output

if __name__ == "__main__":
    from pathlib import Path
    from core.ingest import load_document, chunk_text, embed_chunks
    from core.retrieve import retrieve

    yol = Path(__file__).parent.parent / "data" / "life-changing-daily-habit.txt"
    chunks = chunk_text(load_document(yol))
    doc_vecs = embed_chunks(chunks)

    soru = "What is the capital of France?"
    ilgili_chunklar = retrieve(soru, chunks, doc_vecs, k=3)   # ← önce ara
    context = build_context(ilgili_chunklar)
    print(context)
    
    cevap = generate(soru, context)
    print("Cevap:", cevap.answer)
    print("Kullanılan kaynaklar:", cevap.sources)
    print("Güven:", cevap.reliability)

    