from app.memory.graph import MemoryGraph
from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


def create_memory(content: str) -> ExtractedMemory:

    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.9,
        importance=0.8,
        entities=["Supplier X"],
        topics=["suppliers"],
        source_text=content,
    )


def test_graph_stores_memories():

    graph = MemoryGraph()

    memory_a = create_memory(
        "Supplier X delivered late."
    )

    memory_b = create_memory(
        "Supplier X was rejected."
    )

    graph.add_memory(memory_a)
    graph.add_memory(memory_b)

    assert graph.memory_count() == 2


def test_graph_stores_associations():

    graph = MemoryGraph()

    memory_a = create_memory(
        "Supplier X delivered late."
    )

    memory_b = create_memory(
        "Supplier X was rejected."
    )

    graph.add_memory(memory_a)
    graph.add_memory(memory_b)

    association = Association(
        id="association_001",
        source_id="memory_1",
        target_id="memory_2",
        relationship_type="related_to",
        strength=0.85,
    )

    graph.add_association(association)

    assert graph.association_count() == 1

    connections = graph.get_associations("memory_1")

    assert len(connections) == 1
    assert connections[0].strength == 0.85