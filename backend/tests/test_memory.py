from app.models.memory import Memory


def test_create_memory():
    memory = Memory(
        id="memory_001",
        content="Samuel rejected Supplier X because delivery took 21 days.",
        confidence=0.95,
        importance=0.8,
        source_id="conversation_183"
    )

    assert memory.id == "memory_001"
    assert memory.confidence == 0.95
    assert memory.importance == 0.8
    assert memory.status == "active"