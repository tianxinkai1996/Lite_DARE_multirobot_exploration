"""Direct one-hop contact model.

Decides which robot pairs can exchange messages at a given step, using a
hidden shortest-path distance bound and (optionally) a line-of-sight check.
Only a Boolean contact result is returned; no hidden map state is exposed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


class ContactModel:
    """Produces direct one-hop contact pairs for the current simulator state."""

    def __init__(self, contact_hops=2, require_line_of_sight=True):
        if contact_hops < 0:
            raise ValueError("contact_hops must be non-negative")
        self.contact_hops = int(contact_hops)
        self.require_line_of_sight = bool(require_line_of_sight)

    def can_contact(self, env, robot_i, robot_j):
        """Return True only for close hidden-path distance and unobstructed LoS.

        env's true map and coordinates are used only to emulate a local encounter;
        this method returns a Boolean contact result, never hidden map state.
        """
        pos_i = env.robot_locations[robot_i]
        pos_j = env.robot_locations[robot_j]
        hops = env.hidden_shortest_hops(pos_i, pos_j, self.contact_hops)
        if hops is None:
            return False
        if self.require_line_of_sight and not env.line_is_clear_world(pos_i, pos_j):
            return False
        return True

    def get_contact_pairs(self, env):
        pairs: List[Tuple[int, int]] = []
        for robot_i in range(env.n_agent):
            for robot_j in range(robot_i + 1, env.n_agent):
                if self.can_contact(env, robot_i, robot_j):
                    pairs.append((robot_i, robot_j))
        return pairs
    