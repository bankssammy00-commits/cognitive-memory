from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.memory.graph import MemoryGraph


class BaselineRetriever:

    def __init__(self, graph: MemoryGraph):
        self.graph = graph

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
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
            vectors[1:]
        )[0]

        results = list(
            zip(memory_ids, similarities)
        )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            (memory_id, float(score))
            for memory_id, score in results[:top_k]
        ]