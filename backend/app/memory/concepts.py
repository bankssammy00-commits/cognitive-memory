from collections import defaultdict
from typing import Dict

from app.memory.graph import MemoryGraph


class ConceptIndex:

    STOP_WORDS = {
        "the",
        "and",
        "or",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "within",
        "from",
        "this",
        "that",
    }

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.index: Dict[str, set[str]] = defaultdict(set)

        self._build()

    def normalize(self, word: str) -> str:

        word = word.lower().strip(".,?!:;()[]{}\"'")

        if not word:
            return ""

        # Basic plural normalization.
        if word.endswith("ies"):
            word = word[:-3] + "y"

        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]

        return word

    def tokenize(self, text: str) -> list[str]:

        words = text.lower().split()

        concepts = []

        for word in words:

            concept = self.normalize(word)

            if not concept:
                continue

            if concept in self.STOP_WORDS:
                continue

            if len(concept) <= 2:
                continue

            concepts.append(concept)

        return concepts

    def _build(self):

        for memory_id, memory in self.graph.memories.items():

            # Index individual words from topics.
            for topic in memory.topics:

                concepts = self.tokenize(topic)

                for concept in concepts:

                    self.index[concept].add(
                        memory_id
                    )

            # Also index entities.
            for entity in memory.entities:

                concepts = self.tokenize(entity)

                for concept in concepts:

                    self.index[concept].add(
                        memory_id
                    )

    def find_memories(
        self,
        concepts: list[str],
    ) -> Dict[str, float]:

        results = defaultdict(float)

        for concept in concepts:

            normalized = self.normalize(concept)

            if not normalized:
                continue

            for memory_id in self.index.get(
                normalized,
                set(),
            ):

                results[memory_id] += 1.0

        return dict(results)