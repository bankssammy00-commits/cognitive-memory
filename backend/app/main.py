from fastapi import FastAPI

from app.api.memories import router as memories_router


app = FastAPI(
    title="Cognitive Memory Engine",
    version="0.1.0",
)


app.include_router(memories_router)


@app.get("/")
def root():
    return {
        "name": "Cognitive Memory Engine",
        "version": "0.1.0",
        "status": "running",
    }