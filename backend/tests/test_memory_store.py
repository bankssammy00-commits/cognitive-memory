from app.memory.store import MemoryStore
from app.models.memory import Memory


def test_memory_store():
    store = MemoryStore()

    memory = Memory(
        id="memory_001",
        content="Supplier X delivered in 21 days.",
        confidence=0.95,
        importance=0.8,
        source_id="conversation_001",
    )

    store.add(memory)

    assert store.count() == 1

    retrieved = store.get("memory_001")

    assert retrieved is not None
    assert retrieved.content == "Supplier X delivered in 21 days."


def test_delete_memory():
    store = MemoryStore()

    memory = Memory(
        id="memory_002",
        content="Samuel prefers suppliers with fast delivery.",
    )

    store.add(memory)

    assert store.count() == 1

    deleted = store.delete("memory_002")

    assert deleted is True
    assert store.count() == 0
    assert store.get("memory_002") is None