import numpy as np
from config import settings
import voyageai

vo = voyageai.Client(api_key=settings.voyage_api_key)

def retrieve(question: str, chunks: list[str], doc_vecs, k: int=3) -> list[str]:
    # 1) soruyu query olarak embed et → tek vektör
    query_vec = vo.embed([question], model="voyage-4", input_type="query").embeddings[0]

    # 2) doc vektörlerini matrise çevir (N, 1024)
    doc_matrix = np.array(doc_vecs)

    # 3) benzerlik: (N,1024) @ (1024,) → (N,) skor dizisi
    sims = doc_matrix @ np.array(query_vec)

    # 4) en yüksek k skorun index'i (argsort artan → ters çevir → ilk k)
    top_idx = np.argsort(sims)[::-1][:k]

    # 5) o index'lerdeki chunk METİNLERİNİ döndür
    return [chunks[i] for i in top_idx]


if __name__ == "__main__":
    from pathlib import Path
    from core.ingest import load_document, chunk_text, embed_chunks

    yol = Path(__file__).parent.parent / "data" / "life-changing-daily-habit.txt"
    chunks = chunk_text(load_document(yol))
    doc_vecs = embed_chunks(chunks)

    soru = "What is the daily habit that changes your life?"
    sonuc = retrieve(soru, chunks, doc_vecs, k=3)

    print(f"Soru: {soru}\n")
    for i, c in enumerate(sonuc):
        print(f"--- Top {i+1} ---\n{c[:125]}...\n")