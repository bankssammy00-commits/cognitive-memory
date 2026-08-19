from uuid import uuid4

from fastapi import APIRouter

from app.memory.store import MemoryStore
from app.models.memory import Memory


router = APIRouter()

store = MemoryStore()


@router.post("/memories", response_model=Memory)
def create_memory(memory: Memory) -> Memory:
    if not memory.id:
        memory.id = str(uuid4())

    return store.add(memory)


@router.get("/memories", response_model=list[Memory])
def get_memories() -> list[Memory]:
    return store.get_all()