"""A* shortest-path search over the waypoint mesh.

HOW A* WORKS
------------
A* explores a graph outward from the start, always expanding whichever
frontier node looks most promising. "Most promising" means the lowest:

    f(n) = g(n) + h(n)

    g(n)  what it ACTUALLY cost to reach n from the start
    h(n)  an ESTIMATE of what it will cost to get from n to the goal

Dijkstra's algorithm is the special case h = 0: with no estimate it has
no idea which direction the goal is in, so it expands evenly in all
directions, like a circle growing outward. Adding h pulls the search
toward the goal, so it expands an ellipse instead and touches far fewer
nodes for the same answer.

WHY THE ANSWER IS STILL OPTIMAL
-------------------------------
A* is guaranteed to find the true cheapest path as long as h never
OVERESTIMATES the real remaining cost. That property is called
admissibility.

Here h is straight-line great-circle distance to the destination. No
route between two points can be shorter than the straight line between
them, so h can never overestimate -- it is admissible by construction,
and this search is provably optimal. (If h DID overestimate, A* would
still return a path, just not necessarily the best one.)

Great-circle distance is also *consistent* (it obeys the triangle
inequality), which is the stronger property that lets the algorithm
safely never revisit a node once it has been expanded.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .geo import haversine_nm
from .graph import WaypointGraph

# Type of a function scoring one edge: (graph, from_index, to_index,
# base_cost_nm) -> cost. CP3 will pass one of these in to fold wind and
# restricted airspace into the cost without touching this file.
CostFunction = Callable[[WaypointGraph, int, int, float], float]


@dataclass
class SearchResult:
    """The outcome of one A* run.

    Attributes:
        path: Node indices from start to goal inclusive, empty if none.
        cost: Total path cost (nautical miles, in CP2).
        nodes_expanded: How many nodes were pulled off the frontier.
            Not needed for routing -- it is here to make the efficiency
            of the heuristic measurable against Dijkstra.
    """

    path: List[int]
    cost: float
    nodes_expanded: int

    @property
    def found(self) -> bool:
        return bool(self.path)


def a_star(
    graph: WaypointGraph,
    start: int,
    goal: int,
    cost_function: Optional[CostFunction] = None,
    heuristic_weight: float = 1.0,
    distance_to_cost: Optional[Callable[[float], float]] = None,
) -> SearchResult:
    """Find the cheapest path from `start` to `goal`.

    Args:
        graph: The mesh to search.
        start: Index of the origin node.
        goal: Index of the destination node.
        cost_function: Optional override for edge cost. Receives
            (graph, from_index, to_index, base_distance_nm) and returns a
            cost. Defaults to the plain distance already on the edge.
            Return `math.inf` to make an edge impassable -- that is the
            hook CP3 uses for restricted airspace.
        heuristic_weight: Multiplier on h. Leave at 1.0 for a provably
            optimal result. Above 1.0 makes the search greedier and
            faster but abandons the optimality guarantee -- exposed for
            experimentation, not for production use.
        distance_to_cost: Converts remaining great-circle distance into
            the same units `cost_function` returns. Required whenever
            cost is not distance, and it MUST return a lower bound.

            When cost is time, straight-line distance in nautical miles
            is not comparable to hours, so passing nothing here would
            silently break optimality -- the search would still return a
            route, just not necessarily the cheapest. See
            `cost.make_time_heuristic`.

    Returns:
        A SearchResult. Check `.found` -- a graph can be disconnected.
    """
    if start == goal:
        return SearchResult(path=[start], cost=0.0, nodes_expanded=0)

    goal_node = graph.nodes[goal]

    def heuristic(index: int) -> float:
        """Lower bound on the remaining cost from a node to the goal.

        Straight-line distance is the shortest possible route, so
        converting it with a function that assumes best-case conditions
        can never overestimate -- which is the admissibility condition
        that keeps A* optimal.
        """
        node = graph.nodes[index]
        remaining_nm = haversine_nm(
            node.lat, node.lon, goal_node.lat, goal_node.lon
        )
        return (
            remaining_nm
            if distance_to_cost is None
            else distance_to_cost(remaining_nm)
        )

    # g_score[n] = best known actual cost from start to n.
    g_score: Dict[int, float] = {start: 0.0}

    # came_from[n] = the node we arrived from on the best path to n.
    # Following these backward from the goal reconstructs the route.
    came_from: Dict[int, int] = {}

    # The frontier, ordered by f = g + h. Python's heapq is a min-heap,
    # so the most promising node is always at the front.
    #
    # The tuple carries (f, g, node). Including g as a tiebreaker prefers
    # nodes already closer to the goal when f values tie, and -- more
    # importantly -- prevents heapq from trying to compare node indices
    # in a way that could be ambiguous.
    frontier = [(heuristic(start) * heuristic_weight, 0.0, start)]

    # Nodes already expanded. Because the heuristic is consistent, once a
    # node is expanded its best path is final and it never needs revisiting.
    closed: set = set()

    nodes_expanded = 0

    while frontier:
        _, current_g, current = heapq.heappop(frontier)

        # Stale entry: this node was already expanded via a cheaper route.
        # We push duplicates rather than doing an O(n) decrease-key, which
        # is the standard trade -- heapq has no decrease-key operation.
        if current in closed:
            continue

        if current == goal:
            return SearchResult(
                path=_reconstruct_path(came_from, start, goal),
                cost=current_g,
                nodes_expanded=nodes_expanded,
            )

        closed.add(current)
        nodes_expanded += 1

        for neighbor, base_cost in graph.neighbors(current):
            if neighbor in closed:
                continue

            step_cost = (
                base_cost
                if cost_function is None
                else cost_function(graph, current, neighbor, base_cost)
            )

            # An infinite (or NaN) cost marks an impassable edge, e.g. one
            # crossing restricted airspace in CP3.
            if not math.isfinite(step_cost):
                continue

            tentative_g = current_g + step_cost

            # Only keep this route to the neighbour if it beats whatever
            # we already knew. `float("inf")` as the default means any
            # first-time discovery wins.
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                priority = tentative_g + heuristic(neighbor) * heuristic_weight
                heapq.heappush(frontier, (priority, tentative_g, neighbor))

    # Frontier exhausted without reaching the goal: no path exists.
    return SearchResult(path=[], cost=math.inf, nodes_expanded=nodes_expanded)


def dijkstra(graph: WaypointGraph, start: int, goal: int) -> SearchResult:
    """Dijkstra's algorithm: A* with the heuristic switched off.

    Not used for routing -- it exists to demonstrate what the heuristic
    buys. Both return the same optimal path; compare `nodes_expanded` to
    see how many fewer nodes A* had to touch.
    """
    return a_star(graph, start, goal, heuristic_weight=0.0)


def _reconstruct_path(came_from: Dict[int, int], start: int, goal: int) -> List[int]:
    """Walk the came_from links backward from goal to start.

    Builds the path in reverse, then flips it, which is cheaper than
    inserting at the front of a list repeatedly.
    """
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
