import numpy as np
import voyageai
from anthropic import Anthropic
from datetime import date
from config import settings

client = Anthropic(api_key=settings.anthropic_api_key)
vo = voyageai.Client(api_key=settings.voyage_api_key)

def calculate(expression: str) -> str:
    return str(eval(expression))

def date_of_today() -> str:
    return date.today().isoformat()

# Dispatcher
TOOL_FUNCTIONS = {
    "calculate": calculate,
    "date_of_today": date_of_today
}


tools = [
    {
        "name": "calculate",
        "description": "Bir matematiksel ifadeyi alır ve uygun girdisine göre hesaplayıp sonucunu verir. Matematiksel olmayan ya da anlamsız ifadelerde sonuç vermez.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        },
    },
    {
        "name": "date_of_today",
        "description": "Mevcut günün (bugün) tarih değerini string olarak döndürür",
        "input_schema": {
            "type": "object",
        }
    }
]

def agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    step_count: int = 0

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=settings.max_tokens,
            tools=tools,
            messages=messages
        )

        if response.stop_reason != "tool_use":     # ← çıkış: model tatmin oldu
            return response.content[0].text

        if step_count >= 5:
            return "Maksimum adım sayısına ulaşıldı, işlem durduruldu."

        step_count += 1
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  🔧 {block.name}: {block.input}")
                result = TOOL_FUNCTIONS[block.name](**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    print(agent("5**6-(28/3)"))
    print(agent("Ben 10 Mayıs 2000 tarihinde doğdum. Şu anda kaç yaşındayım?"))
    print(agent("Ben 8 Kasım 2026 tarihinde askerlikten terhis olacağım. Terhis olmama kaç günüm kaldı?"))