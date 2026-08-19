from app.memory.association_discovery import (
    AssociationDiscovery,
)

from app.memory.graph import MemoryGraph

from app.models.memory_extraction import (
    ExtractedMemory,
)


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


def test_discovers_shared_entity():

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

    discovery = AssociationDiscovery(
        graph
    )

    associations = discovery.discover()

    assert len(associations) > 0

    association = associations[0]

    assert association.source_id == "memory_1"

    assert association.target_id == "memory_2"


def test_ignores_unrelated_memories():

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
            "My laptop has 8GB RAM.",
            ["Laptop"],
            ["computer", "hardware"],
        )
    )

    discovery = AssociationDiscovery(
        graph
    )

    associations = discovery.discover()

    assert len(associations) == 0