## Önemli Notlar

### 🧾 Batch vs Async — Cheatsheet

**Tek cümlelik tanımlar**

- **Batch** = *çok veriyi TEK istekte* gönder → **round-trip sayısını** azaltır.
- **Async** = *çok isteği EŞZAMANLI* gönder → **bekleme sürelerini üst üste bindirir** (bloklamaz).

İkisi rakip değil, **farklı eksende** iki kaldıraç. Biri "kaç istek", diğeri "istekler nasıl beklenir".

**Zihinsel model 🍽️**

Restoranda 10 kişilik masasın:

- **Naif (döngü):** Garson her kişinin siparişini ayrı ayrı mutfağa götürüp bekler, döner, sonrakini alır. 10 gidiş-dönüş.
- **Batch:** Garson 10 siparişi tek kağıda yazıp **bir kez** mutfağa götürür. 1 gidiş-dönüş.
- **Async:** 10 garson aynı anda 10 masaya bakar; biri beklerken diğerleri çalışır. Bekleme süreleri çakışır.

**Ne zaman hangisi?**

| Durum | Çözüm |
|---|---|
| API tek çağrıda liste kabul ediyor (Voyage `embed`, OpenAI embeddings) | **Batch** |
| Her çağrı bağımsız + API liste kabul etmiyor (LLM `messages.create`, 100 ayrı soru) | **Async** (`asyncio.gather`) |
| Devasa hacim, batch limitini aşıyorsun (50.000 chunk, limit 1000) | **İkisi**: 50 batch → `gather` ile paralel |
| CPU işi (numpy hesap, saf Python) | **Hiçbiri** — async I/O beklemesini gizler, hesabı hızlandırmaz |

**Kod kalıpları**

Batch (embedding — Faz 1):

```python
# 1 istek, 1000 metin
vecs = vo.embed(chunks, model="voyage-4", input_type="document").embeddings
```

Async (bağımsız LLM çağrıları):

```python
import asyncio

async def sor(q):
    return await client.messages.create(...)   # await = "bekle ama bloklamadan"

# 100 soru eşzamanlı — Promise.all([...]) karşılığı
cevaplar = await asyncio.gather(*[sor(q) for q in sorular])
```

Batch + Async birlikte (limit aşımı):

```python
batches = [chunks[i:i+1000] for i in range(0, len(chunks), 1000)]
sonuclar = await asyncio.gather(*[embed_batch(b) for b in batches])
```

**JS köprüsü 🌉**

- **Batch** ≈ tek `fetch(url, {body: JSON.stringify(items)})` — array gönderirsin.
- **Async** ≈ `await Promise.all(items.map(x => fetch(...)))` — N eşzamanlı istek.
- `asyncio.gather` ≈ `Promise.all`. `await` ≈ `await`. Python'da fark: `async def` fonksiyonu ancak bir **event loop** içinde (`asyncio.run(...)`) çalışır; JS'te loop her zaman gizlice oradadır.

**⚠️ Karıştırma tuzağı**

"Batch = paralel/async" **DEĞİL.** Batch tek, senkron bir HTTP isteği — paralellik yok, sadece az round-trip. Async ise paralellik/eşzamanlılık.

**Neden ikisi de maliyet/performans meselesi 💰**

- **Batch:** az round-trip = az gecikme + bazı API'lerde **batch indirimi** (%50'ye kadar).
- **Async:** toplam süre = en yavaş istek (sıralıda = tüm sürelerin toplamı). 100 çağrı × 1sn: sıralı 100sn, async ~1-2sn.