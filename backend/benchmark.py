from app.memory.activation import ActivationEngine
from app.memory.baseline import BaselineRetriever
from app.memory.retrieval import RetrievalEngine
from app.memory.graph import MemoryGraph
from app.models.association import Association
from app.models.memory_extraction import ExtractedMemory


def make_memory(
    content: str,
    entities: list[str],
    topics: list[str],
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


# ============================================================
# BUILD MEMORY GRAPH
# ============================================================

graph = MemoryGraph()


graph.add_memory(
    make_memory(
        "I require suppliers to deliver within 14 days.",
        ["User"],
        ["supplier requirements", "delivery"],
    )
)

graph.add_memory(
    make_memory(
        "Supplier Alpha usually delivers in 9 days.",
        ["Supplier Alpha"],
        ["suppliers", "delivery"],
    )
)

graph.add_memory(
    make_memory(
        "Supplier Beta usually takes 22 days.",
        ["Supplier Beta"],
        ["suppliers", "delivery"],
    )
)

graph.add_memory(
    make_memory(
        "I rejected Supplier Beta because they were too slow.",
        ["Supplier Beta"],
        ["suppliers", "delivery", "rejection"],
    )
)

graph.add_memory(
    make_memory(
        "Supplier Alpha increased their prices.",
        ["Supplier Alpha"],
        ["suppliers", "pricing"],
    )
)


# ============================================================
# EXISTING ASSOCIATIONS
# ============================================================

graph.add_association(
    Association(
        id="a1",
        source_id="memory_1",
        target_id="memory_2",
        relationship_type="related_to",
        strength=0.7,
    )
)

graph.add_association(
    Association(
        id="a2",
        source_id="memory_1",
        target_id="memory_3",
        relationship_type="related_to",
        strength=0.7,
    )
)

graph.add_association(
    Association(
        id="a3",
        source_id="memory_3",
        target_id="memory_4",
        relationship_type="caused",
        strength=0.9,
    )
)

graph.add_association(
    Association(
        id="a4",
        source_id="memory_2",
        target_id="memory_5",
        relationship_type="related_to",
        strength=0.8,
    )
)


# ============================================================
# NEW SEMANTIC CONSTRAINT RELATIONSHIPS
# ============================================================

# The user's 14-day requirement constrains Alpha.

graph.add_association(
    Association(
        id="a5",
        source_id="memory_1",
        target_id="memory_2",
        relationship_type="constrains",
        strength=0.95,
    )
)


# The user's 14-day requirement also constrains Beta.

graph.add_association(
    Association(
        id="a6",
        source_id="memory_1",
        target_id="memory_3",
        relationship_type="constrains",
        strength=0.95,
    )
)


# ============================================================
# MEMORY ENGINES
# ============================================================

activation = ActivationEngine(graph)

cognitive = RetrievalEngine(
    graph,
    activation,
)

baseline = BaselineRetriever(graph)


# ============================================================
# GRAPH PROPAGATION DIAGNOSTIC
# ============================================================

print("\nGRAPH PROPAGATION TEST")
print("======================\n")

activated = activation.activate(
    "memory_1",
    initial_activation=1.0,
    max_steps=3,
)

for memory_id, score in activated.items():

    memory = graph.get_memory(memory_id)

    print(
        f"{memory_id} | "
        f"{score:.4f} | "
        f"{memory.content}"
    )


# ============================================================
# RETRIEVAL BENCHMARK
# ============================================================

query = "Which supplier should I consider?"


print("\nBASELINE RESULTS")
print("================\n")

baseline_results = baseline.retrieve(
    query,
    top_k=5,
)

for memory_id, score in baseline_results:

    memory = graph.get_memory(memory_id)

    print(
        f"{memory_id} | "
        f"{score:.4f} | "
        f"{memory.content}"
    )


print("\nCOGNITIVE MEMORY RESULTS")
print("========================\n")

cognitive_results = cognitive.retrieve(
    query,
    top_k=5,
)

for memory_id, score in cognitive_results:

    memory = graph.get_memory(memory_id)

    print(
        f"{memory_id} | "
        f"{score:.4f} | "
        f"{memory.content}"
    )