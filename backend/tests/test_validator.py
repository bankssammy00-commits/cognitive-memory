from app.memory.validator import MemoryValidator
from app.models.memory_extraction import ExtractedMemory


def test_valid_memory():

    validator = MemoryValidator()

    memory = ExtractedMemory(
        content="User requires suppliers to deliver within 14 days.",
        memory_type="constraint",
        confidence=0.95,
        importance=0.8,
        entities=["Supplier"],
        topics=["suppliers"],
        source_text="I need suppliers to deliver within 14 days.",
    )

    assert validator.validate(memory) is True


def test_low_confidence_memory_is_rejected():

    validator = MemoryValidator()

    memory = ExtractedMemory(
        content="User might like Supplier X.",
        memory_type="preference",
        confidence=0.3,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers"],
        source_text="Maybe Supplier X is good.",
    )

    assert validator.validate(memory) is False