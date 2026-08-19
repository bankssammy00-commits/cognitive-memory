from app.memory.concepts import ConceptIndex
from app.memory.graph import MemoryGraph
from app.models.memory_extraction import ExtractedMemory


def make_memory(content, topics):

    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.9,
        importance=0.8,
        entities=[],
        topics=topics,
        source_text=content,
    )


def test_concept_index():

    graph = MemoryGraph()

    graph.add_memory(
        make_memory(
            "Supplier Alpha delivers in 9 days.",
            ["suppliers", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "The football match ended 2-1.",
            ["sports"],
        )
    )

    index = ConceptIndex(graph)

    results = index.find_memories(
        ["suppliers"]
    )

    assert "memory_1" in results
    assert "memory_2" not in results