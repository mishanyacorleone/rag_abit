import uvicorn
from fastapi import FastAPI
from app.api import api
from app.api import admin
from app.config.config import get_settings

settings = get_settings()
settings.print_config()


app = FastAPI(
    title="МГТУ бот", 
    version="1.0", 
    description="API для ответов на вопросы абитуриентов + управление данными",
    lifespan=api.lifespan
)

app.include_router(api.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": "Отвечаем на вопросы абитуриентов. Документация: /docs",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/generate/",
            "admin": "/api/admin/qdrant/stats"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app=f"{__name__}:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )