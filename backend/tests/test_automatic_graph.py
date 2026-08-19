from app.memory.activation import ActivationEngine
from app.memory.association_discovery import AssociationDiscovery
from app.memory.graph import MemoryGraph
from app.memory.retrieval import RetrievalEngine
from app.models.memory_extraction import ExtractedMemory


def make_memory(
    content,
    entities,
    topics,
):
    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.95,
        importance=0.8,
        entities=entities,
        topics=topics,
        source_text=content,
    )


def test_automatic_graph_and_retrieval():

    graph = MemoryGraph()

    graph.add_memory(
        make_memory(
            "I require suppliers to deliver within 14 days.",
            ["User"],
            ["supplier", "delivery", "requirement"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Alpha usually delivers in 9 days.",
            ["Supplier Alpha"],
            ["supplier", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Beta usually takes 22 days.",
            ["Supplier Beta"],
            ["supplier", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "I rejected Supplier Beta because they were too slow.",
            ["Supplier Beta"],
            ["supplier", "delivery", "rejection"],
        )
    )

    discovery = AssociationDiscovery(
        graph,
        min_strength=0.20,
    )

    associations = discovery.build_graph()

    assert len(associations) > 0

    activation = ActivationEngine(
        graph
    )

    retrieval = RetrievalEngine(
        graph,
        activation,
    )

    results = retrieval.retrieve(
        "Which supplier meets my delivery requirement?",
        top_k=4,
    )

    assert len(results) > 0

    returned_ids = [
        memory_id
        for memory_id, score in results
    ]

    assert "memory_2" in returned_ids


def test_no_manual_associations_exist():

    graph = MemoryGraph()

    graph.add_memory(
        make_memory(
            "Supplier Alpha delivers in 9 days.",
            ["Supplier Alpha"],
            ["supplier", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Alpha increased prices.",
            ["Supplier Alpha"],
            ["supplier", "pricing"],
        )
    )

    assert len(graph.associations) == 0

    discovery = AssociationDiscovery(
        graph
    )

    discovery.build_graph()

    assert len(graph.associations) > 0