import uvicorn
from fastapi import FastAPI
from app.api import api


app = FastAPI(title="МГТУ бот", version="1.0", lifespan=api.lifespan)
app.include_router(api.router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Отвечаем на вопросы абитуриентов. Документация: /docs"}


if __name__ == "__main__":
    uvicorn.run(app=f"{__name__}:app", host="0.0.0.0", port=8081, reload=True)