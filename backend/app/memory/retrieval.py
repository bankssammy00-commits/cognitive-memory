from datetime import datetime
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.memory.activation import ActivationEngine
from app.memory.concepts import ConceptIndex
from app.memory.graph import MemoryGraph


class RetrievalEngine:

    def __init__(
        self,
        graph: MemoryGraph,
        activation_engine: ActivationEngine,
    ):
        self.graph = graph
        self.activation_engine = activation_engine
        self.concept_index = ConceptIndex(graph)

    # ---------------------------------------------------------
    # 1. Lexical retrieval
    # ---------------------------------------------------------

    def lexical_candidates(
        self,
        query: str,
    ) -> List[Tuple[str, float]]:

        if not self.graph.memories:
            return []

        memory_ids = list(self.graph.memories.keys())

        documents = [
            self.graph.memories[memory_id].content
            for memory_id in memory_ids
        ]

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [query] + documents
        )

        similarities = cosine_similarity(
            vectors[0:1],
            vectors[1:],
        )[0]

        results = list(
            zip(memory_ids, similarities)
        )

        results.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (memory_id, float(score))
            for memory_id, score in results
            if score > 0
        ]

    # ---------------------------------------------------------
    # 2. Concept retrieval
    # ---------------------------------------------------------

    def concept_candidates(
        self,
        query: str,
    ) -> dict[str, float]:

        words = query.lower().split()

        concepts = [
            word.strip(".,?!")
            for word in words
            if len(word) > 3
        ]

        return self.concept_index.find_memories(
            concepts
        )

    # ---------------------------------------------------------
    # 3. Query intent
    # ---------------------------------------------------------

    def detect_query_intent(
        self,
        query: str,
    ) -> dict[str, bool]:

        query = query.lower()

        return {
            "causal": any(
                phrase in query
                for phrase in [
                    "why",
                    "reason",
                    "because",
                    "caused",
                    "cause",
                    "what led to",
                    "what happened",
                ]
            ),

            "rejection": any(
                phrase in query
                for phrase in [
                    "rejected",
                    "reject",
                    "rejection",
                    "declined",
                    "discarded",
                    "didn't choose",
                    "did not choose",
                    "not choose",
                ]
            ),

            "constraint": any(
                phrase in query
                for phrase in [
                    "requirement",
                    "requirements",
                    "require",
                    "must",
                    "within",
                    "under",
                    "maximum",
                    "minimum",
                    "limit",
                ]
            ),

            "preference": any(
                phrase in query
                for phrase in [
                    "prefer",
                    "preference",
                    "favorite",
                    "favourite",
                    "like",
                    "usually choose",
                ]
            ),

            "pricing": any(
                phrase in query
                for phrase in [
                    "price",
                    "pricing",
                    "cost",
                    "expensive",
                    "cheap",
                    "cheaper",
                    "increase",
                    "increased",
                ]
            ),
        }

    # ---------------------------------------------------------
    # 4. Intent relevance
    # ---------------------------------------------------------

    def intent_score(
        self,
        memory_id: str,
        intent: dict[str, bool],
    ) -> float:

        memory = self.graph.get_memory(memory_id)

        if memory is None:
            return 0.0

        content = memory.content.lower()

        score = 0.0

        # -----------------------------------------------------
        # Causal reasoning
        # -----------------------------------------------------

        if intent["causal"]:

            causal_terms = [
                "because",
                "reason",
                "caused",
                "due to",
                "therefore",
                "result",
                "led to",
            ]

            if any(
                term in content
                for term in causal_terms
            ):
                score += 0.25

        # -----------------------------------------------------
        # Rejection reasoning
        # -----------------------------------------------------

        if intent["rejection"]:

            rejection_terms = [
                "rejected",
                "reject",
                "rejection",
                "declined",
                "discarded",
                "didn't choose",
                "did not choose",
            ]

            if any(
                term in content
                for term in rejection_terms
            ):
                score += 0.35

        # -----------------------------------------------------
        # Constraint reasoning
        # -----------------------------------------------------

        if intent["constraint"]:

            constraint_terms = [
                "require",
                "requirement",
                "requirements",
                "must",
                "within",
                "maximum",
                "minimum",
                "limit",
            ]

            if any(
                term in content
                for term in constraint_terms
            ):
                score += 0.20

        # -----------------------------------------------------
        # Preference reasoning
        # -----------------------------------------------------

        if intent["preference"]:

            preference_terms = [
                "prefer",
                "preference",
                "favorite",
                "favourite",
                "like",
                "usually choose",
            ]

            if any(
                term in content
                for term in preference_terms
            ):
                score += 0.20

        # -----------------------------------------------------
        # Pricing reasoning
        # -----------------------------------------------------

        if intent["pricing"]:

            pricing_terms = [
                "price",
                "pricing",
                "cost",
                "expensive",
                "cheap",
                "cheaper",
                "increase",
                "increased",
            ]

            if any(
                term in content
                for term in pricing_terms
            ):
                score += 0.20

        return score

    # ---------------------------------------------------------
    # 5. Temporal relevance
    # ---------------------------------------------------------

    def temporal_score(
        self,
        memory_id: str,
    ) -> float:

        memory = self.graph.get_memory(memory_id)

        if memory is None:
            return 0.0

        valid_from = getattr(
            memory,
            "valid_from",
            None,
        )

        valid_until = getattr(
            memory,
            "valid_until",
            None,
        )

        # No temporal information.
        # Return neutral rather than a positive bonus.
        if valid_from is None and valid_until is None:
            return 0.0

        now = datetime.utcnow()

        if (
            valid_from is not None
            and valid_from > now
        ):
            return -1.0

        if (
            valid_until is not None
            and valid_until < now
        ):
            return -1.0

        return 1.0

    # ---------------------------------------------------------
    # 6. Relationship weighting
    # ---------------------------------------------------------

    def relationship_weight(
        self,
        relationship_type: str,
    ) -> float:

        relationship_type = relationship_type.lower()

        weights = {
            "causal": 1.35,
            "caused": 1.35,

            "constrains": 1.20,
            "constraint": 1.20,

            "supports": 1.05,
            "preferred": 1.05,

            "related_to": 0.75,

            "contradicts": -1.00,
            "negative": -1.00,
            "opposes": -1.00,
        }

        return weights.get(
            relationship_type,
            0.70,
        )

    # ---------------------------------------------------------
    # 7. Graph propagation
    # ---------------------------------------------------------

    def propagate(
        self,
        candidate_scores: dict[str, float],
    ) -> dict[str, float]:

        propagated_scores: dict[str, float] = {}

        for memory_id, score in candidate_scores.items():

            if score <= 0:
                continue

            activated = self.activation_engine.activate(
                memory_id,
                initial_activation=score,
                max_steps=3,
            )

            for activated_id, activation in activated.items():

                if activated_id == memory_id:
                    continue

                associations = self.graph.get_associations(
                    memory_id
                )

                relationship_multiplier = 1.0

                for association in associations:

                    connected = (
                        (
                            association.source_id == memory_id
                            and association.target_id == activated_id
                        )
                        or
                        (
                            association.target_id == memory_id
                            and association.source_id == activated_id
                        )
                    )

                    if connected:
                        relationship_multiplier = (
                            self.relationship_weight(
                                association.relationship_type
                            )
                        )
                        break

                weighted_activation = (
                    activation
                    * relationship_multiplier
                )

                propagated_scores[activated_id] = (
                    propagated_scores.get(
                        activated_id,
                        0.0,
                    )
                    + weighted_activation
                )

        return propagated_scores

    # ---------------------------------------------------------
    # 8. Retrieve
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):

        if top_k <= 0:
            return []

        # -----------------------------------------------------
        # Lexical signal
        # -----------------------------------------------------

        lexical = self.lexical_candidates(query)

        lexical_scores = {
            memory_id: score
            for memory_id, score in lexical
        }

        # -----------------------------------------------------
        # Concept signal
        # -----------------------------------------------------

        conceptual = self.concept_candidates(query)

        # -----------------------------------------------------
        # Candidate generation
        # -----------------------------------------------------

        candidate_ids = set(
            lexical_scores.keys()
        )

        candidate_ids.update(
            conceptual.keys()
        )

        if not candidate_ids:
            return []

        # -----------------------------------------------------
        # Normalize concept scores
        # -----------------------------------------------------

        max_concept = max(
            conceptual.values(),
            default=0.0,
        )

        if max_concept > 0:

            conceptual_scores = {
                memory_id: score / max_concept
                for memory_id, score in conceptual.items()
            }

        else:

            conceptual_scores = {}

        # -----------------------------------------------------
        # Query intent
        # -----------------------------------------------------

        intent = self.detect_query_intent(query)

        # -----------------------------------------------------
        # Direct relevance
        # -----------------------------------------------------

        direct_scores: dict[str, float] = {}

        for memory_id in candidate_ids:

            lexical_score = lexical_scores.get(
                memory_id,
                0.0,
            )

            concept_score = conceptual_scores.get(
                memory_id,
                0.0,
            )

            semantic_score = (
                lexical_score * 0.65
                + concept_score * 0.35
            )

            intent_relevance = self.intent_score(
                memory_id,
                intent,
            )

            direct_scores[memory_id] = (
                semantic_score
                + intent_relevance
            )

        # -----------------------------------------------------
        # Temporal signal
        # -----------------------------------------------------

        temporal_scores = {
            memory_id: self.temporal_score(memory_id)
            for memory_id in candidate_ids
        }

        # -----------------------------------------------------
        # Graph propagation
        # -----------------------------------------------------

        propagated_scores = self.propagate(
            direct_scores
        )

        # -----------------------------------------------------
        # Final ranking
        # -----------------------------------------------------

        results = []

        for memory_id in candidate_ids:

            direct = direct_scores.get(
                memory_id,
                0.0,
            )

            propagated = propagated_scores.get(
                memory_id,
                0.0,
            )

            temporal = temporal_scores.get(
                memory_id,
                0.0,
            )

            # Direct evidence is strongest.
            final_score = (
                direct * 0.70
                + propagated * 0.25
                + temporal * 0.05
            )

            # Temporal invalidity should suppress a memory.
            if temporal < 0:
                final_score *= 0.25

            # Keep scores bounded.
            final_score = max(
                0.0,
                min(
                    1.0,
                    final_score,
                ),
            )

            results.append(
                (
                    memory_id,
                    float(final_score),
                )
            )

        # -----------------------------------------------------
        # Deterministic ranking
        # -----------------------------------------------------

        results.sort(
            key=lambda x: (
                -x[1],
                x[0],
            )
        )

        return results[:top_k]