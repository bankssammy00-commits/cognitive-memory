from app.memory.graph import MemoryGraph


class AssociationLearner:
    """
    Learns from retrieval feedback by adjusting association strength.

    Positive feedback strengthens an association.
    Negative feedback weakens an association.

    Association strength is always kept in the range [0.0, 1.0].
    """

    def __init__(
        self,
        graph: MemoryGraph,
        learning_rate: float = 0.10,
        min_strength: float = 0.0,
        max_strength: float = 1.0,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")

        if not 0.0 <= min_strength <= 1.0:
            raise ValueError("min_strength must be between 0 and 1")

        if not 0.0 <= max_strength <= 1.0:
            raise ValueError("max_strength must be between 0 and 1")

        if min_strength > max_strength:
            raise ValueError(
                "min_strength cannot be greater than max_strength"
            )

        self.graph = graph
        self.learning_rate = learning_rate
        self.min_strength = min_strength
        self.max_strength = max_strength

    # ---------------------------------------------------------
    # Strength adjustment
    # ---------------------------------------------------------

    def _clamp(
        self,
        value: float,
    ) -> float:
        return max(
            self.min_strength,
            min(
                self.max_strength,
                value,
            ),
        )

    def reinforce(
        self,
        association_id: str,
        reward: float = 1.0,
    ) -> bool:
        """
        Strengthen an association using positive feedback.

        Returns True if the association was found and updated.
        """

        if reward < 0:
            raise ValueError(
                "reinforce reward must be non-negative"
            )

        association = self._find_association(
            association_id
        )

        if association is None:
            return False

        adjustment = (
            self.learning_rate
            * reward
            * (self.max_strength - association.strength)
        )

        association.strength = self._clamp(
            association.strength + adjustment
        )

        return True

    def weaken(
        self,
        association_id: str,
        penalty: float = 1.0,
    ) -> bool:
        """
        Weaken an association using negative feedback.

        Returns True if the association was found and updated.
        """

        if penalty < 0:
            raise ValueError(
                "weaken penalty must be non-negative"
            )

        association = self._find_association(
            association_id
        )

        if association is None:
            return False

        adjustment = (
            self.learning_rate
            * penalty
            * (
                association.strength
                - self.min_strength
            )
        )

        association.strength = self._clamp(
            association.strength - adjustment
        )

        return True

    # ---------------------------------------------------------
    # Unified learning interface
    # ---------------------------------------------------------

    def learn(
        self,
        association_id: str,
        reward: float,
    ) -> bool:
        """
        Apply feedback in the range [-1.0, 1.0].

        Positive reward -> reinforce.
        Negative reward -> weaken.
        Zero -> no change.
        """

        if not -1.0 <= reward <= 1.0:
            raise ValueError(
                "reward must be between -1.0 and 1.0"
            )

        if reward > 0:
            return self.reinforce(
                association_id,
                reward=reward,
            )

        if reward < 0:
            return self.weaken(
                association_id,
                penalty=abs(reward),
            )

        return self._find_association(
            association_id
        ) is not None

    # ---------------------------------------------------------
    # Association lookup
    # ---------------------------------------------------------

    def _find_association(
        self,
        association_id: str,
    ):
        for association in self.graph.associations.values():
            if association.id == association_id:
                return association

        return None