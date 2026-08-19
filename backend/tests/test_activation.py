from app.memory.activation import ActivationEngine
from app.memory.graph import MemoryGraph
from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


def memory(content: str):
    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.9,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers"],
        source_text=content,
    )


def test_activation_spreads():

    graph = MemoryGraph()

    graph.add_memory(
        memory("Supplier X delivered late.")
    )

    graph.add_memory(
        memory("Supplier X was rejected.")
    )

    graph.add_memory(
        memory("Supplier X violated the delivery requirement.")
    )

    graph.add_association(
        Association(
            id="a1",
            source_id="memory_1",
            target_id="memory_2",
            relationship_type="related_to",
            strength=0.8,
        )
    )

    graph.add_association(
        Association(
            id="a2",
            source_id="memory_2",
            target_id="memory_3",
            relationship_type="related_to",
            strength=0.6,
        )
    )

    engine = ActivationEngine(graph)

    result = engine.activate(
        "memory_1",
        max_steps=2,
    )

    assert "memory_2" in result
    assert "memory_3" in result

    assert result["memory_2"] > result["memory_3"]