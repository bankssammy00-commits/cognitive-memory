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


def test_does_not_create_false_causal_relationship():

    graph = MemoryGraph()

    graph.add_memory(
        make_memory(
            "Supplier Alpha usually delivers in 9 days.",
            ["Supplier Alpha"],
            ["supplier", "delivery"],
        )
    )

    graph.add_memory(
        make_memory(
            "Supplier Beta was rejected because deliveries took 22 days.",
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

    false_causal_relationship = False

    for association in associations:

        pair = {
            association.source_id,
            association.target_id,
        }

        if pair == {"memory_1", "memory_3"}:

            if association.relationship_type == "causal":
                false_causal_relationship = True

    assert not false_causal_relationship