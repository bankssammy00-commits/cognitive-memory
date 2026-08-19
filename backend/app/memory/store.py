from typing import Dict, List, Optional

from app.models.memory import Memory


class MemoryStore:
    def __init__(self):
        self._memories: Dict[str, Memory] = {}

    def add(self, memory: Memory) -> Memory:
        self._memories[memory.id] = memory
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        return self._memories.get(memory_id)

    def get_all(self) -> List[Memory]:
        return list(self._memories.values())

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._memories:
            return False

        del self._memories[memory_id]
        return True

    def count(self) -> int:
        return len(self._memories)