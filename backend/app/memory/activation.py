from collections import defaultdict
from typing import Dict

from app.memory.graph import MemoryGraph


class ActivationEngine:

    def __init__(
        self,
        graph: MemoryGraph,
        decay: float = 0.7,
        threshold: float = 0.05,
    ):
        self.graph = graph
        self.decay = decay
        self.threshold = threshold

    def activate(
        self,
        starting_memory_id: str,
        initial_activation: float = 1.0,
        max_steps: int = 3,
    ) -> Dict[str, float]:

        # Total activation accumulated by each memory.
        activation: Dict[str, float] = {
            starting_memory_id: initial_activation
        }

        # Memories activated during the current wave.
        frontier: Dict[str, float] = {
            starting_memory_id: initial_activation
        }

        # Prevent immediate cycles from repeatedly
        # amplifying the same memories.
        visited = {
            starting_memory_id
        }

        for step in range(max_steps):

            next_frontier: Dict[str, float] = {}

            for memory_id, current_activation in frontier.items():

                associations = self.graph.get_associations(
                    memory_id
                )

                for association in associations:

                    if association.source_id == memory_id:
                        target = association.target_id
                    else:
                        target = association.source_id

                    # Don't immediately reactivate memories
                    # we've already processed.
                    if target in visited:
                        continue

                    propagated = (
                        current_activation
                        * association.strength
                        * self.decay
                    )

                    if propagated < self.threshold:
                        continue

                    # Keep the strongest activation path.
                    previous = activation.get(
                        target,
                        0.0,
                    )

                    activation[target] = max(
                        previous,
                        propagated,
                    )

                    next_frontier[target] = max(
                        next_frontier.get(
                            target,
                            0.0,
                        ),
                        propagated,
                    )

            # Mark newly activated memories as visited.
            visited.update(
                next_frontier.keys()
            )

            frontier = next_frontier

            if not frontier:
                break

        return dict(activation)