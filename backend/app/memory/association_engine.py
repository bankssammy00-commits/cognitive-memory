from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.memory_extraction import ExtractedMemory


class AssociationEngine:

    def calculate_similarity(
        self,
        memory_a: ExtractedMemory,
        memory_b: ExtractedMemory,
    ) -> float:

        text_a = f"{memory_a.content} {' '.join(memory_a.topics)}"
        text_b = f"{memory_b.content} {' '.join(memory_b.topics)}"

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [text_a, text_b]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(similarity)

    def shared_entity_score(
        self,
        memory_a: ExtractedMemory,
        memory_b: ExtractedMemory,
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

        shared = entities_a.intersection(entities_b)

        return len(shared) / max(
            len(entities_a.union(entities_b)),
            1
        )

    def shared_topic_score(
        self,
        memory_a: ExtractedMemory,
        memory_b: ExtractedMemory,
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

        shared = topics_a.intersection(topics_b)

        return len(shared) / max(
            len(topics_a.union(topics_b)),
            1
        )

    def calculate_association_strength(
        self,
        memory_a: ExtractedMemory,
        memory_b: ExtractedMemory,
    ) -> float:

        semantic = self.calculate_similarity(
            memory_a,
            memory_b
        )

        entity = self.shared_entity_score(
            memory_a,
            memory_b
        )

        topic = self.shared_topic_score(
            memory_a,
            memory_b
        )

        strength = (
            semantic * 0.5
            + entity * 0.3
            + topic * 0.2
        )

        return round(
            min(max(strength, 0.0), 1.0),
            4
        )