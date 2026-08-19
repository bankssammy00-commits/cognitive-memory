from app.memory.association_discovery import AssociationDiscovery
from app.memory.graph import MemoryGraph
from app.models.memory_extraction import ExtractedMemory


def make_memory(content, entities, topics):

    return ExtractedMemory(
        content=content,
        memory_type="event",
        confidence=0.95,
        importance=0.8,
        entities=entities,
        topics=topics,
        source_text=content,
    )


def test_association_quality():

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

    graph.add_memory(
        make_memory(
            "Supplier Alpha increased their prices.",
            ["Supplier Alpha"],
            ["supplier", "pricing"],
        )
    )

    discovery = AssociationDiscovery(
        graph,
        min_strength=0.20,
    )

    associations = discovery.build_graph()

    assert len(associations) > 0

    # Important relationships that SHOULD exist.

    pairs = {
        (
            association.source_id,
            association.target_id,
        )
        for association in associations
    }

    assert (
        ("memory_1", "memory_2") in pairs
        or
        ("memory_2", "memory_1") in pairs
    )

    assert (
        ("memory_3", "memory_4") in pairs
        or
        ("memory_4", "memory_3") in pairs
    )