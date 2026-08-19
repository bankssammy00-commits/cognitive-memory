from app.models.memory_extraction import ExtractedMemory


def test_extracted_memory():
    memory = ExtractedMemory(
        content="Supplier X repeatedly delivered late.",
        memory_type="event",
        confidence=0.9,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers", "delivery"],
        source_text="I stopped using Supplier X because they kept delivering late.",
    )

    assert memory.content == "Supplier X repeatedly delivered late."
    assert memory.memory_type == "event"
    assert "Supplier X" in memory.entities
    assert "delivery" in memory.topics
    assert memory.source_text != ""