import ast
import operator
from datetime import date

from anthropic import Anthropic

from config import settings
from core.rag import build_context
from core.retrieve import retrieve

client = Anthropic(api_key=settings.anthropic_api_key)


# ----------------------------------------------------------------------------
# Güvenli hesap makinesi
#
# eval() KULLANMIYORUZ: expression modelin ürettiği, kullanıcının yönlendirdiği
# bir string. eval onu Python KODU olarak çalıştırır ->
# "__import__('os').system('cat .env')" gibi bir girdi API key'leri sızdırır.
# Bunun yerine ifadeyi parse edip ağaçta SADECE izin verdiğimiz düğümleri
# değerlendiriyoruz (whitelist). Fonksiyon çağrısı, import, değişken adı:
# hiç değerlendirilmeden reddedilir.
# ----------------------------------------------------------------------------
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    """AST düğümünü özyinelemeli değerlendirir. İzin listesi dışındaki her şey hata."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Sadece sayı sabitleri desteklenir.")

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("Desteklenmeyen operatör.")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # DoS koruması: 10**10**10 gibi bir ifade belleği tüketir.
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 10**6):
            raise ValueError("Üs işlemi çok büyük.")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError("Desteklenmeyen tekli operatör.")
        return op(_eval_node(node.operand))

    raise ValueError("İfade sadece sayı ve aritmetik operatör içerebilir.")


def calculate(expression: str) -> str:
    """Tool fonksiyonu: hata fırlatmaz, MESAJ döndürür.

    Sebep: dönen string modele geri gider. Model 'Hesaplanamadı: ...' okuyunca
    ifadeyi düzeltip tekrar deneyebilir. Exception fırlatsaydık tüm istek çökerdi.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        return f"Hesaplanamadı: {e}"


def date_of_today() -> str:
    return date.today().isoformat()


def make_search_tool(chunks: list[str], doc_vecs):
    """FABRİKA: chunks/doc_vecs'i 'içine kapatıp' tek argümanlı bir tool döndürür.

    Model sadece 'query' verir; index'i closure taşır.
    """

    def search_documents(query: str) -> str:
        sonuc, skorlar = retrieve(query, chunks, doc_vecs, k=3)
        if not sonuc:
            # Modele boş string değil, ANLAMLI bir cümle dön.
            return "Dokümanda bu konuyla ilgili kaynak bulunamadı."
        return build_context(sonuc)

    return search_documents  # parantez YOK: fonksiyonun kendisini döndürüyoruz


def build_tool_functions(chunks: list[str], doc_vecs) -> dict:
    """Dispatcher. search_documents index'e ihtiyaç duyduğu için çalışma anında kurulur."""
    return {
        "calculate": calculate,
        "date_of_today": date_of_today,
        "search_documents": make_search_tool(chunks, doc_vecs),
    }


# ----------------------------------------------------------------------------
# Tool şemaları (modelin gördüğü kullanım kılavuzu)
# ----------------------------------------------------------------------------
tools = [
    {
        "name": "search_documents",
        "description": (
            "Kullanıcının yüklediği doküman koleksiyonunda semantik arama yapar ve "
            "en alakalı pasajları <kaynak id=\"N\"> etiketleri içinde döndürür.\n"
            "NE ZAMAN KULLAN: Doküman içeriğine dair her soruda — 'ne diyor', "
            "'nasıl yapılır', 'hangi alışkanlık' gibi bilgi soruları. Emin değilsen KULLAN.\n"
            "NE ZAMAN KULLANMA: Saf aritmetik veya tarih hesabı için (onlar için "
            "calculate ve date_of_today var).\n"
            "İPUCU: query'yi kullanıcının cümlesinin aynısı yapmak zorunda değilsin; "
            "arama için en uygun anahtar ifadeyi yaz."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Dokümanda aranacak konu veya soru.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Bir aritmetik ifadeyi hesaplar. Sadece sayılar ve + - * / // % ** "
            "operatörlerini destekler; fonksiyon çağrısı veya değişken kabul etmez.\n"
            "NE ZAMAN KULLAN: Herhangi bir aritmetik işlem gerektiğinde — kafadan "
            "hesaplama yapma, her zaman bu tool'u kullan.\n"
            "NE ZAMAN KULLANMA: İfade matematiksel değilse veya zaten bilinen tek bir "
            "sayıysa (örn. calculate('69') anlamsızdır)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Aritmetik ifade, örn. '5**6-(28/3)'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "date_of_today",
        "description": (
            "Bugünün tarihini YYYY-AA-GG formatında döndürür.\n"
            "NE ZAMAN KULLAN: 'bugün', 'şu an', 'kaç yaşındayım', 'kaç gün kaldı' gibi "
            "güncel tarihe dayanan her hesapta — tarihi tahmin etme."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = (
    "Sen kullanıcının dokümanları üzerinde çalışan bir asistansın.\n"
    "Doküman içeriğine dair bilgiyi ASLA kendi bilginden verme; önce search_documents "
    "ile ara, sadece dönen <kaynak> etiketlerindeki bilgiyi kullan.\n"
    "Kaynaklarda cevap yoksa bunu açıkça söyle, uydurma.\n"
    "Hesap gerektiren her adımda calculate kullan."
)


# ----------------------------------------------------------------------------
# Agent döngüsü
# ----------------------------------------------------------------------------
def agent(question: str, chunks: list[str], doc_vecs, max_adim: int = 5) -> str:
    tool_functions = build_tool_functions(chunks, doc_vecs)
    messages = [{"role": "user", "content": question}]
    step_count = 0

    # 1. tur: aramayı GARANTİ et. Sonraki turlarda serbest bırak (yoksa sonsuz döngü).
    tool_choice = {"type": "tool", "name": "search_documents"}

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
        )
        tool_choice = {"type": "auto"}  # ← ilk turdan sonra model karar versin

        if response.stop_reason != "tool_use":  # çıkış: model tatmin oldu
            return response.content[0].text

        if step_count >= max_adim:
            return "Maksimum adım sayısına ulaşıldı, işlem durduruldu."

        step_count += 1
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  🔧 {block.name}: {block.input}")
                result = tool_functions[block.name](**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    from pathlib import Path
    from core.ingest import load_document, chunk_text, embed_chunks

    # --- Güvenlik testi: bunlar ÇALIŞMAMALI ---
    print("🔒 Güvenlik testleri:")
    for kotu in ["__import__('os').system('ls')", "open('.env').read()", "10**10**10"]:
        print(f"   {kotu!r:40} -> {calculate(kotu)}")
    print(f"   {'5**6-(28/3)':40} -> {calculate('5**6-(28/3)')}")

    yol = Path(__file__).parent.parent / "data" / "life-changing-daily-habit.txt"
    chunks = chunk_text(load_document(yol))
    doc_vecs = embed_chunks(chunks)

    sorular = [
        "What is the daily habit that changes your life?",           # sadece RAG
        "Bu alışkanlığı 30 gün uygularsam toplam kaç saat eder?",    # RAG + calculate
        "Bugünden 2027 yılbaşına kaç gün var?",                      # date + calculate
    ]
    for s in sorular:
        print(f"\n{'=' * 60}\n❓ {s}\n")
        print(agent(s, chunks, doc_vecs))
