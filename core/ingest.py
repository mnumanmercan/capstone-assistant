from pathlib import Path


def load_document(path: Path) -> str:
    """Tek bir .txt dosyasını oku, içeriğini string döndür."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Dar try: SADECE okuma işini sardık. Ne yapacağını çağıran karar versin.
        raise ValueError(f"Doküman bulunamadı: {path}")


def chunk_text(text: str) -> list[str]:
    """Metni paragraf sınırlarından (çift satır boşluk) parçalara böl."""
    parcalar = text.split("\n\n")
    # Her parçanın kenar boşluğunu temizle, tamamen boş olanları at:
    return [p.strip() for p in parcalar if p.strip()]


if __name__ == "__main__":
    yol = Path(__file__).parent.parent / "data" / "life-changing-daily-habit.txt"
    metin = load_document(yol)
    chunklar = chunk_text(metin)
    print(f"Toplam {len(chunklar)} chunk bulundu.\n")
    for i, c in enumerate(chunklar):
        print(f"--- Chunk {i} ({len(c)} karakter) ---")
        print(c[:80], "...\n")   # ilk 80 karakter önizleme