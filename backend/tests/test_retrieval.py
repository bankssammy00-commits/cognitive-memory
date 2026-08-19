from app.memory.activation import ActivationEngine
from app.memory.graph import MemoryGraph
from app.memory.retrieval import RetrievalEngine
from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


def make_memory(
    content: str,
    entity: str,
    topic: str,
):
    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.95,
        importance=0.8,
        entities=[entity],
        topics=[topic],
        source_text=content,
    )


def test_retrieval():

    graph = MemoryGraph()

    graph.add_memory(
        make_memory(
            "Supplier Alpha delivers in 9 days.",
            "Supplier Alpha",
            "suppliers",
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Alpha increased prices.",
            "Supplier Alpha",
            "suppliers",
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Beta delivers in 22 days.",
            "Supplier Beta",
            "suppliers",
        )
    )

    graph.add_association(
        Association(
            id="a1",
            source_id="memory_1",
            target_id="memory_2",
            relationship_type="related_to",
            strength=0.9,
        )
    )

    activation = ActivationEngine(graph)

    retrieval = RetrievalEngine(
        graph,
        activation,
    )

    results = retrieval.retrieve(
        "Which supplier delivers quickly?",
        top_k=3,
    )

    assert len(results) > 0

    returned_ids = [
        memory_id
        for memory_id, score in results
    ]

    assert "memory_1" in returned_ids