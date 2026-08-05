"""The waypoint mesh graph that A* searches over.

WHAT A "MESH GRAPH" IS HERE
---------------------------
Real flight plans are written over published airways -- named routes
connecting named fixes, like a road network. That data is not in
OurAirports (see the README's scope section), so this project synthesises
a stand-in: take every navaid in the region and connect the ones near
enough to each other to be a sensible leg. The result is a graph, and
finding a route becomes finding a shortest path through it.

Nodes are navaids, plus the origin and destination airports so the path
has somewhere to start and end. Intermediate airports are deliberately
NOT nodes: real routes are defined navaid-to-navaid and no dispatcher
routes a flight over an airport it is not landing at.

BUILT PER REQUEST, NOT ONCE GLOBALLY
------------------------------------
A single mesh covering the whole US would have ~1,300 navaids and
millions of candidate edges, most of them irrelevant to any given
flight. Instead the graph is built fresh for each request over just the
region between origin and destination -- a few hundred nodes, built in
milliseconds. That is why `build_mesh` takes an already-filtered list
rather than reaching for the full dataset itself.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .geo import haversine_nm
from .models import Waypoint

# Two nodes are connected if they are within this far apart.
#
# Tuned by measurement, not guesswork. Sweeping this value on KJFK->KLAX
# (2,146 nm direct, 1,604 navaids in region):
#
#     radius   waypoints   route nm   % of direct   nodes expanded
#         75          39     2197.2        102.4%              796
#        100          29     2167.2        101.0%              634
#        150          21     2154.9        100.4%              436
#        200          15     2147.7        100.1%              173
#        300          10     2146.2        100.0%               67
#
# A tight radius forces many short hops, each slightly off the great
# circle, and the deviations accumulate -- 75 nm was actually WORSE than
# CP1's naive sampling on that route. A loose radius converges on the
# direct line but produces a flight plan with almost no waypoints in it,
# which is useless for navigation even though the distance is optimal.
#
# 150 nm sits at the knee: within half a percent of optimal, with leg
# lengths in the range a real en-route flight plan uses.
DEFAULT_RADIUS_NM = 150.0

# ...but a radius alone can strand a node in a sparse region (over water,
# over desert) with no neighbours at all, and a stranded node means A*
# reports "no route found". So every node also keeps at least this many
# nearest neighbours regardless of how far away they are.
DEFAULT_MIN_NEIGHBORS = 6


@dataclass
class WaypointGraph:
    """An undirected graph of waypoints with distance-weighted edges.

    Nodes are stored in a list and referred to everywhere by integer
    index rather than by object. Integers are cheap to hash and compare,
    which matters inside A*'s inner loop, and it keeps the adjacency
    structure compact.

    Attributes:
        nodes: The waypoints, indexed by position.
        adjacency: adjacency[i] is a list of (neighbour_index, cost_nm).
    """

    nodes: List[Waypoint]
    adjacency: List[List[Tuple[int, float]]] = field(default_factory=list)

    def index_of(self, ident: str) -> int:
        """Find a node's index by identifier. Raises KeyError if absent."""
        for index, node in enumerate(self.nodes):
            if node.ident == ident:
                return index
        raise KeyError(f"{ident} is not a node in this graph")

    def neighbors(self, index: int) -> List[Tuple[int, float]]:
        """(neighbour_index, cost_nm) pairs reachable from `index`."""
        return self.adjacency[index]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of undirected edges.

        Every edge is stored twice (once in each endpoint's adjacency
        list), so halve the total to count real connections.
        """
        return sum(len(edges) for edges in self.adjacency) // 2


def build_mesh(
    waypoints: Sequence[Waypoint],
    radius_nm: float = DEFAULT_RADIUS_NM,
    min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
) -> WaypointGraph:
    """Connect waypoints into an undirected mesh.

    THE EDGE RULE (hybrid: radius with a k-nearest floor)
    -----------------------------------------------------
    For each node:
      1. Connect to every node within `radius_nm`.
      2. If that produced fewer than `min_neighbors` connections, top it
         up with the nearest nodes regardless of distance.

    Rule 1 gives a dense, realistic mesh wherever navaids are dense.
    Rule 2 guarantees the graph never fragments into disconnected islands
    just because one node sits in an empty patch -- which would make A*
    report "no route found" for a perfectly reasonable flight.

    Edge cost is great-circle distance for now. CP3 replaces it with
    time-given-wind, and sets it to infinity for edges crossing
    restricted airspace. A* itself will not change -- only this number.

    Complexity is O(n^2): every node is compared against every other. At
    a few hundred nodes per request that is a few tens of thousands of
    distance computations, which runs in milliseconds. It would NOT be
    acceptable for a global mesh, which is why the caller filters the
    region first.
    """
    node_list = list(waypoints)
    count = len(node_list)

    # `pairs` deduplicates: an edge found from both endpoints is stored
    # once here, then expanded into both adjacency lists at the end. This
    # is what keeps the graph undirected even though the k-nearest floor
    # is applied asymmetrically (node A may need B as a fallback
    # neighbour while B, in a dense area, does not need A).
    pairs: Dict[Tuple[int, int], float] = {}

    for i in range(count):
        # Distance to every other node, computed once and reused by both
        # the radius test and the nearest-neighbour fallback.
        distances = [
            (
                haversine_nm(
                    node_list[i].lat, node_list[i].lon,
                    node_list[j].lat, node_list[j].lon,
                ),
                j,
            )
            for j in range(count)
            if j != i
        ]

        within_radius = [(dist, j) for dist, j in distances if dist <= radius_nm]

        # Sparse-region fallback: take the closest `min_neighbors` nodes
        # even though they are beyond the radius.
        if len(within_radius) < min_neighbors:
            distances.sort()
            within_radius = distances[:min_neighbors]

        for dist, j in within_radius:
            # Canonical key ordering so (i,j) and (j,i) collapse to one.
            pairs[(min(i, j), max(i, j))] = dist

    _bridge_components(node_list, pairs)

    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(count)]
    for (i, j), dist in pairs.items():
        adjacency[i].append((j, dist))
        adjacency[j].append((i, dist))

    return WaypointGraph(nodes=node_list, adjacency=adjacency)


def _bridge_components(
    node_list: Sequence[Waypoint], pairs: Dict[Tuple[int, int], float]
) -> None:
    """Join disconnected clusters with their shortest connecting edges.

    Mutates `pairs` in place.

    WHY THE k-NEAREST FLOOR IS NOT ENOUGH
    -------------------------------------
    The floor guarantees no node is *isolated*, but that is a weaker
    property than the graph being *connected*. Two dense clusters with an
    ocean between them each satisfy the floor internally and never link
    to each other. Measured on KJFK->EGLL: 695 nodes, of which only 350
    were reachable from the origin -- North America and Europe formed
    separate islands, and A* correctly reported no route.

    Navaids are ground stations, and there is no ground mid-Atlantic, so
    no radius setting can fix this. Instead: find the clusters, and join
    them with the shortest edge available between each pair. This is
    Kruskal's idea applied to whole components -- repeatedly add the
    cheapest edge that merges two of them, until one remains.

    For an already-connected graph (every land route) this is a no-op, so
    normal routing is unaffected.
    """
    count = len(node_list)
    if count < 2:
        return

    # Union-find, so "which cluster is this node in" stays cheap as we merge.
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]  # path compression
            node = parent[node]
        return node

    for i, j in pairs:
        parent[find(i)] = find(j)

    # Group nodes by cluster.
    clusters: Dict[int, List[int]] = {}
    for node in range(count):
        clusters.setdefault(find(node), []).append(node)

    while len(clusters) > 1:
        # Find the shortest edge joining any two distinct clusters.
        roots = list(clusters)
        best = None
        for a_index in range(len(roots)):
            for b_index in range(a_index + 1, len(roots)):
                for i in clusters[roots[a_index]]:
                    for j in clusters[roots[b_index]]:
                        dist = haversine_nm(
                            node_list[i].lat, node_list[i].lon,
                            node_list[j].lat, node_list[j].lon,
                        )
                        if best is None or dist < best[0]:
                            best = (dist, i, j, roots[a_index], roots[b_index])

        dist, i, j, root_a, root_b = best
        pairs[(min(i, j), max(i, j))] = dist

        # Merge the two clusters.
        clusters[root_a].extend(clusters.pop(root_b))


def connected_component_size(graph: WaypointGraph, start: int) -> int:
    """How many nodes are reachable from `start`.

    A diagnostic, not part of routing. If this comes back well below
    `graph.node_count`, the mesh has fragmented into islands and the
    radius or min_neighbors setting is too tight for the region.
    """
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for neighbor, _ in graph.neighbors(current):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen)


def nearest_node(graph: WaypointGraph, lat: float, lon: float) -> int:
    """Index of the graph node closest to a coordinate.

    Uses a heap rather than a full sort because only the single best
    result is wanted.
    """
    return heapq.nsmallest(
        1,
        range(graph.node_count),
        key=lambda i: haversine_nm(lat, lon, graph.nodes[i].lat, graph.nodes[i].lon),
    )[0]
