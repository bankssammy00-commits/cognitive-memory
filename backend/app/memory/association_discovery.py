from itertools import combinations

from app.memory.graph import MemoryGraph
from app.models.association import Association


class AssociationDiscovery:

    def __init__(
        self,
        graph: MemoryGraph,
        min_strength: float = 0.30,
    ):
        self.graph = graph
        self.min_strength = min_strength

    # ---------------------------------------------------------
    # ENTITY OVERLAP
    # ---------------------------------------------------------

    def entity_similarity(
        self,
        memory_a,
        memory_b,
    ) -> float:

        entities_a = {
            entity.lower()
            for entity in memory_a.entities
        }

        entities_b = {
            entity.lower()
            for entity in memory_b.entities
        }

        if not entities_a or not entities_b:
            return 0.0

        intersection = (
            entities_a & entities_b
        )

        union = (
            entities_a | entities_b
        )

        return len(intersection) / len(union)

    # ---------------------------------------------------------
    # TOPIC OVERLAP
    # ---------------------------------------------------------

    def topic_similarity(
        self,
        memory_a,
        memory_b,
    ) -> float:

        topics_a = {
            topic.lower()
            for topic in memory_a.topics
        }

        topics_b = {
            topic.lower()
            for topic in memory_b.topics
        }

        if not topics_a or not topics_b:
            return 0.0

        intersection = (
            topics_a & topics_b
        )

        union = (
            topics_a | topics_b
        )

        return len(intersection) / len(union)

    # ---------------------------------------------------------
    # CONTENT OVERLAP
    # ---------------------------------------------------------

    def word_similarity(
        self,
        memory_a,
        memory_b,
    ) -> float:

        words_a = {
            word.lower().strip(".,!?")
            for word in memory_a.content.split()
            if len(word) > 3
        }

        words_b = {
            word.lower().strip(".,!?")
            for word in memory_b.content.split()
            if len(word) > 3
        }

        if not words_a or not words_b:
            return 0.0

        intersection = (
            words_a & words_b
        )

        union = (
            words_a | words_b
        )

        return len(intersection) / len(union)

    # ---------------------------------------------------------
    # RELATIONSHIP TYPE
    # ---------------------------------------------------------

    def relationship_type(
        self,
        memory_a,
        memory_b,
    ) -> str:

        text_a = memory_a.content.lower()
        text_b = memory_b.content.lower()

        combined = text_a + " " + text_b

        # Causal language
        causal_words = [
            "because",
            "therefore",
            "caused",
            "due to",
            "resulted",
            "reason",
            "rejected",
        ]

        if any(
            word in combined
            for word in causal_words
        ):
            return "causal"

        # Constraint language
        constraint_words = [
            "require",
            "requires",
            "must",
            "need",
            "within",
            "limit",
            "maximum",
            "minimum",
        ]

        if any(
            word in combined
            for word in constraint_words
        ):
            return "constrains"

        return "related_to"

    # ---------------------------------------------------------
    # ASSOCIATION STRENGTH
    # ---------------------------------------------------------

    def calculate_strength(
        self,
        memory_a,
        memory_b,
    ) -> float:

        entity_score = (
            self.entity_similarity(
                memory_a,
                memory_b,
            )
        )

        topic_score = (
            self.topic_similarity(
                memory_a,
                memory_b,
            )
        )

        word_score = (
            self.word_similarity(
                memory_a,
                memory_b,
            )
        )

        strength = (
            entity_score * 0.50
            + topic_score * 0.30
            + word_score * 0.20
        )

        return min(
            1.0,
            strength,
        )

    # ---------------------------------------------------------
    # DISCOVER
    # ---------------------------------------------------------

    def discover(self):

        memories = list(
            self.graph.memories.items()
        )

        discovered = []

        for (
            (id_a, memory_a),
            (id_b, memory_b),
        ) in combinations(
            memories,
            2,
        ):

            strength = (
                self.calculate_strength(
                    memory_a,
                    memory_b,
                )
            )

            if strength < self.min_strength:
                continue

            relationship = (
                self.relationship_type(
                    memory_a,
                    memory_b,
                )
            )

            association_id = (
                f"auto_{id_a}_{id_b}"
            )

            association = Association(
                id=association_id,
                source_id=id_a,
                target_id=id_b,
                relationship_type=relationship,
                strength=strength,
            )

            discovered.append(
                association
            )

        return discovered

    # ---------------------------------------------------------
    # APPLY TO GRAPH
    # ---------------------------------------------------------

    def build_graph(self):

        associations = (
            self.discover()
        )

        for association in associations:

            self.graph.add_association(
                association
            )

        return associations