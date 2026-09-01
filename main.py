from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def get_status():
    return {"status": "ok"}
