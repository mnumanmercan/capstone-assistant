import numpy as np
import voyageai

from config import settings

vo = voyageai.Client(api_key=settings.voyage_api_key)


def retrieve(
    question: str,
    chunks: list[str],
    doc_vecs,
    k: int = 3,
) -> tuple[list[str], list[float]]:
    """En alakalı k chunk'ı VE benzerlik skorlarını döndürür.

    Skorlar artık atılmıyor: reliability hesabında kullanacağız.
    """
    # 1) soruyu query olarak embed et -> tek vektör
    query_vec = vo.embed([question], model="voyage-4", input_type="query").embeddings[0]

    # 2) doc vektörlerini matrise çevir (N, 1024)
    doc_matrix = np.array(doc_vecs)

    # 3) benzerlik: (N,1024) @ (1024,) -> (N,) skor dizisi
    #    Voyage vektörleri length-1 normalize -> cosine = düz nokta çarpımı
    sims = doc_matrix @ np.array(query_vec)

    # 4) en yüksek k skorun index'i
    top_idx = np.argsort(sims)[::-1][:k]

    # 5) metinler + skorlar (np.float32 -> float, JSON serialize edilebilsin)
    return [chunks[i] for i in top_idx], [float(sims[i]) for i in top_idx]


if __name__ == "__main__":
    from pathlib import Path
    from core.ingest import load_document, chunk_text, embed_chunks

    yol = Path(__file__).parent.parent / "data" / "life-changing-daily-habit.txt"
    chunks = chunk_text(load_document(yol))
    doc_vecs = embed_chunks(chunks)

    soru = "What is the daily habit that changes your life?"
    sonuc, skorlar = retrieve(soru, chunks, doc_vecs, k=3)

    print(f"Soru: {soru}\n")
    for i, (c, s) in enumerate(zip(sonuc, skorlar)):
        print(f"--- Top {i + 1} (skor: {s:.3f}) ---\n{c[:125]}...\n")
