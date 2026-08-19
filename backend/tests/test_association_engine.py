from app.memory.association_engine import AssociationEngine
from app.models.memory_extraction import ExtractedMemory


def test_related_memories_have_strong_association():

    engine = AssociationEngine()

    memory_a = ExtractedMemory(
        content="Supplier X delivered late.",
        memory_type="event",
        confidence=0.95,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers", "delivery"],
        source_text="Supplier X delivered late.",
    )

    memory_b = ExtractedMemory(
        content="Supplier X was rejected because of slow delivery.",
        memory_type="decision",
        confidence=0.95,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers", "delivery"],
        source_text="Supplier X was rejected because of slow delivery.",
    )

    strength = engine.calculate_association_strength(
        memory_a,
        memory_b,
    )

    assert strength > 0.3


def test_unrelated_memories_have_weaker_association():

    engine = AssociationEngine()

    memory_a = ExtractedMemory(
        content="Supplier X delivered late.",
        memory_type="event",
        confidence=0.95,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers", "delivery"],
        source_text="Supplier X delivered late.",
    )

    memory_b = ExtractedMemory(
        content="The football match ended 2-1.",
        memory_type="event",
        confidence=0.95,
        importance=0.5,
        entities=["Football"],
        topics=["sports"],
        source_text="The football match ended 2-1.",
    )

    strength = engine.calculate_association_strength(
        memory_a,
        memory_b,
    )

    assert strength < 0.3