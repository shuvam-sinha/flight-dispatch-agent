import math
import unittest

from flight_dispatch.graph import WaypointGraph, build_mesh
from flight_dispatch.models import Navaid
from flight_dispatch.search import a_star, dijkstra


def navaid(ident: str, lat: float, lon: float) -> Navaid:
    return Navaid(ident=ident, name=ident, navaid_type="VOR", lat=lat, lon=lon)


def manual_graph(nodes, edges) -> WaypointGraph:
    """Build a graph with hand-specified edges, bypassing build_mesh.

    Lets a test assert on exact costs rather than on whatever the mesh
    rule happens to produce.
    """
    adjacency = [[] for _ in nodes]
    for i, j, cost in edges:
        adjacency[i].append((j, cost))
        adjacency[j].append((i, cost))
    return WaypointGraph(nodes=list(nodes), adjacency=adjacency)


class TestAStarBasics(unittest.TestCase):
    def test_start_equals_goal(self):
        graph = build_mesh([navaid("A", 40.0, -90.0), navaid("B", 40.5, -90.0)])
        result = a_star(graph, 0, 0)
        self.assertEqual(result.path, [0])
        self.assertEqual(result.cost, 0.0)

    def test_direct_neighbours(self):
        graph = build_mesh([navaid("A", 40.0, -90.0), navaid("B", 40.5, -90.0)])
        result = a_star(graph, 0, 1)
        self.assertTrue(result.found)
        self.assertEqual(result.path, [0, 1])
        self.assertAlmostEqual(result.cost, 30.0, delta=0.5)

    def test_path_starts_at_start_and_ends_at_goal(self):
        graph = build_mesh(
            [navaid(f"N{i}", 40.0 + i * 0.5, -90.0) for i in range(6)],
            radius_nm=40,
            min_neighbors=1,
        )
        result = a_star(graph, 0, 5)
        self.assertEqual(result.path[0], 0)
        self.assertEqual(result.path[-1], 5)

    def test_reports_no_path_when_graph_is_disconnected(self):
        nodes = [navaid("A", 40.0, -90.0), navaid("B", 40.1, -90.0),
                 navaid("C", 50.0, -90.0), navaid("D", 50.1, -90.0)]
        # No edges between the two clusters.
        graph = manual_graph(nodes, [(0, 1, 6.0), (2, 3, 6.0)])
        result = a_star(graph, 0, 3)
        self.assertFalse(result.found)
        self.assertEqual(result.path, [])
        self.assertEqual(result.cost, math.inf)


class TestAStarOptimality(unittest.TestCase):
    """A* must return the genuinely cheapest path, not merely a path."""

    def test_prefers_the_cheaper_of_two_routes(self):
        # A -> B -> D costs 10; A -> C -> D costs 4.
        nodes = [navaid("A", 40.0, -90.0), navaid("B", 40.1, -90.0),
                 navaid("C", 40.1, -89.9), navaid("D", 40.2, -89.95)]
        graph = manual_graph(nodes, [
            (0, 1, 5.0), (1, 3, 5.0),
            (0, 2, 2.0), (2, 3, 2.0),
        ])
        result = a_star(graph, 0, 3)
        self.assertEqual(result.path, [0, 2, 3])
        self.assertAlmostEqual(result.cost, 4.0)

    def test_prefers_many_cheap_hops_over_one_expensive_edge(self):
        nodes = [navaid(f"N{i}", 40.0 + i * 0.01, -90.0) for i in range(5)]
        graph = manual_graph(nodes, [
            (0, 4, 100.0),                                    # one big hop
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),  # four small
        ])
        result = a_star(graph, 0, 4)
        self.assertEqual(result.path, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(result.cost, 4.0)

    def test_agrees_with_dijkstra(self):
        # The heuristic must not change the answer, only the work done.
        nodes = [navaid(f"N{i}", 40.0 + (i % 5) * 0.3, -90.0 + (i // 5) * 0.3)
                 for i in range(25)]
        graph = build_mesh(nodes, radius_nm=40, min_neighbors=2)

        astar_result = a_star(graph, 0, 24)
        dijkstra_result = dijkstra(graph, 0, 24)

        self.assertAlmostEqual(astar_result.cost, dijkstra_result.cost, places=6)

    def test_heuristic_reduces_work(self):
        """The whole point of A* over Dijkstra."""
        nodes = [navaid(f"N{i}", 40.0 + (i % 8) * 0.25, -90.0 + (i // 8) * 0.25)
                 for i in range(64)]
        graph = build_mesh(nodes, radius_nm=30, min_neighbors=2)

        astar_result = a_star(graph, 0, 63)
        dijkstra_result = dijkstra(graph, 0, 63)

        self.assertLess(astar_result.nodes_expanded, dijkstra_result.nodes_expanded)


class TestCostFunction(unittest.TestCase):
    """The hook CP3 uses for wind and restricted airspace."""

    def test_custom_cost_can_redirect_the_route(self):
        nodes = [navaid("A", 40.0, -90.0), navaid("B", 40.1, -90.0),
                 navaid("C", 40.1, -89.9), navaid("D", 40.2, -89.95)]
        graph = manual_graph(nodes, [
            (0, 1, 5.0), (1, 3, 5.0),
            (0, 2, 2.0), (2, 3, 2.0),
        ])
        # Make anything touching node 2 wildly expensive.
        def penalise_c(_graph, from_index, to_index, base):
            return base * 100 if 2 in (from_index, to_index) else base

        result = a_star(graph, 0, 3, cost_function=penalise_c)
        self.assertEqual(result.path, [0, 1, 3])

    def test_infinite_cost_blocks_an_edge(self):
        """How CP3 will exclude edges crossing restricted airspace."""
        nodes = [navaid("A", 40.0, -90.0), navaid("B", 40.1, -90.0),
                 navaid("C", 40.1, -89.9), navaid("D", 40.2, -89.95)]
        graph = manual_graph(nodes, [
            (0, 1, 5.0), (1, 3, 5.0),
            (0, 2, 2.0), (2, 3, 2.0),
        ])
        def block_via_c(_graph, from_index, to_index, base):
            return math.inf if 2 in (from_index, to_index) else base

        result = a_star(graph, 0, 3, cost_function=block_via_c)
        self.assertEqual(result.path, [0, 1, 3])
        self.assertAlmostEqual(result.cost, 10.0)

    def test_blocking_every_edge_yields_no_route(self):
        nodes = [navaid("A", 40.0, -90.0), navaid("B", 40.1, -90.0)]
        graph = manual_graph(nodes, [(0, 1, 5.0)])
        result = a_star(graph, 0, 1, cost_function=lambda *_: math.inf)
        self.assertFalse(result.found)


if __name__ == "__main__":
    unittest.main()
