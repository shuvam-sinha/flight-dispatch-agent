import unittest

from flight_dispatch.geo import haversine_nm
from flight_dispatch.graph import (
    build_mesh,
    connected_component_size,
    nearest_node,
)
from flight_dispatch.models import Navaid


def navaid(ident: str, lat: float, lon: float) -> Navaid:
    return Navaid(ident=ident, name=ident, navaid_type="VOR", lat=lat, lon=lon)


def chain(count: int, spacing_deg: float = 0.5) -> list:
    """`count` navaids in a straight line, evenly spaced along a meridian."""
    return [navaid(f"N{i}", 40.0 + i * spacing_deg, -90.0) for i in range(count)]


class TestMeshConstruction(unittest.TestCase):
    def test_every_waypoint_becomes_a_node(self):
        graph = build_mesh(chain(5))
        self.assertEqual(graph.node_count, 5)
        self.assertEqual([n.ident for n in graph.nodes], ["N0", "N1", "N2", "N3", "N4"])

    def test_nearby_nodes_are_connected(self):
        # 0.5 deg apart == 30 nm, comfortably inside a 75 nm radius.
        graph = build_mesh(chain(3), radius_nm=75, min_neighbors=1)
        neighbors = {j for j, _ in graph.neighbors(0)}
        self.assertIn(1, neighbors)

    def test_distant_nodes_are_not_connected_by_radius(self):
        # N0 to N4 is 2 degrees == 120 nm, outside a 45 nm radius. With
        # min_neighbors=1 the fallback cannot pull it in either, since N1
        # is closer.
        graph = build_mesh(chain(5), radius_nm=45, min_neighbors=1)
        self.assertNotIn(4, {j for j, _ in graph.neighbors(0)})

    def test_edge_cost_is_great_circle_distance(self):
        graph = build_mesh(chain(2), radius_nm=75, min_neighbors=1)
        _, cost = graph.neighbors(0)[0]
        expected = haversine_nm(40.0, -90.0, 40.5, -90.0)
        self.assertAlmostEqual(cost, expected, places=6)

    def test_graph_is_undirected(self):
        graph = build_mesh(chain(6), radius_nm=75, min_neighbors=2)
        for i in range(graph.node_count):
            for j, _ in graph.neighbors(i):
                self.assertIn(
                    i, {k for k, _ in graph.neighbors(j)},
                    f"edge {i}->{j} exists but {j}->{i} does not",
                )

    def test_no_self_loops(self):
        graph = build_mesh(chain(4))
        for i in range(graph.node_count):
            self.assertNotIn(i, {j for j, _ in graph.neighbors(i)})

    def test_edge_count_halves_the_adjacency_total(self):
        graph = build_mesh(chain(2), radius_nm=75, min_neighbors=1)
        # One undirected edge, stored in both adjacency lists.
        self.assertEqual(graph.edge_count, 1)

    def test_single_node_graph_has_no_edges(self):
        graph = build_mesh([navaid("ONLY", 40.0, -90.0)])
        self.assertEqual(graph.node_count, 1)
        self.assertEqual(graph.edge_count, 0)

    def test_empty_input(self):
        graph = build_mesh([])
        self.assertEqual(graph.node_count, 0)
        self.assertEqual(graph.edge_count, 0)


class TestMinNeighborsFloor(unittest.TestCase):
    """The k-nearest fallback is what stops sparse regions fragmenting."""

    def test_isolated_node_still_gets_connected(self):
        # Three clustered navaids plus one 600 nm away, far outside any
        # sensible radius. Without the floor it would be unreachable.
        points = [
            navaid("A", 40.0, -90.0),
            navaid("B", 40.2, -90.0),
            navaid("C", 40.4, -90.0),
            navaid("LONE", 50.0, -90.0),
        ]
        graph = build_mesh(points, radius_nm=50, min_neighbors=2)
        self.assertGreaterEqual(len(graph.neighbors(3)), 2)

    def test_floor_keeps_the_graph_in_one_piece(self):
        points = [
            navaid("A", 40.0, -90.0),
            navaid("B", 40.2, -90.0),
            navaid("FAR1", 55.0, -90.0),
            navaid("FAR2", 55.2, -90.0),
        ]
        graph = build_mesh(points, radius_nm=40, min_neighbors=3)
        self.assertEqual(connected_component_size(graph, 0), graph.node_count)

    def test_graph_is_connected_even_with_the_floor_disabled(self):
        # Two clusters 900 nm apart, radius far too small to span them,
        # and min_neighbors=1 so the floor does nothing. Component
        # bridging must still join them.
        points = [
            navaid("A", 40.0, -90.0),
            navaid("B", 40.2, -90.0),
            navaid("FAR1", 55.0, -90.0),
            navaid("FAR2", 55.2, -90.0),
        ]
        graph = build_mesh(points, radius_nm=40, min_neighbors=1)
        self.assertEqual(connected_component_size(graph, 0), graph.node_count)


class TestComponentBridging(unittest.TestCase):
    """The floor stops nodes being isolated; it does NOT make the graph
    connected. Two dense clusters either side of an ocean each satisfy the
    floor internally and never link up. Bridging fixes that."""

    def two_clusters(self) -> list:
        near = [navaid(f"A{i}", 40.0 + i * 0.1, -90.0) for i in range(5)]
        far = [navaid(f"B{i}", 40.0 + i * 0.1, -60.0) for i in range(5)]
        return near + far

    def test_distant_clusters_are_joined(self):
        graph = build_mesh(self.two_clusters(), radius_nm=50, min_neighbors=2)
        self.assertEqual(connected_component_size(graph, 0), graph.node_count)

    def test_bridge_uses_the_shortest_available_pair(self):
        graph = build_mesh(self.two_clusters(), radius_nm=50, min_neighbors=2)
        # The closest cross-cluster pair is A4 (40.4N) to B4 (40.4N):
        # same latitude, so the shortest span between the two groups.
        bridge_ends = set()
        for i in range(graph.node_count):
            for j, _ in graph.neighbors(i):
                a, b = graph.nodes[i].ident, graph.nodes[j].ident
                if a[0] != b[0]:  # an A-node joined to a B-node
                    bridge_ends.add(frozenset({a, b}))
        self.assertEqual(bridge_ends, {frozenset({"A4", "B4"})})

    def test_connected_graph_is_left_alone(self):
        # Everything already within radius: bridging must add nothing.
        dense = chain(5, spacing_deg=0.1)
        with_bridging = build_mesh(dense, radius_nm=75, min_neighbors=2)
        self.assertEqual(with_bridging.edge_count, 10)  # still complete, no extras

    def test_three_clusters_all_join(self):
        points = (
            [navaid(f"A{i}", 40.0 + i * 0.1, -90.0) for i in range(3)]
            + [navaid(f"B{i}", 40.0 + i * 0.1, -70.0) for i in range(3)]
            + [navaid(f"C{i}", 40.0 + i * 0.1, -50.0) for i in range(3)]
        )
        graph = build_mesh(points, radius_nm=50, min_neighbors=2)
        self.assertEqual(connected_component_size(graph, 0), graph.node_count)

    def test_dense_region_is_unaffected_by_the_floor(self):
        # Everything already within radius, so the floor changes nothing.
        dense = build_mesh(chain(5, spacing_deg=0.1), radius_nm=75, min_neighbors=2)
        self.assertEqual(dense.edge_count, 10)  # complete graph on 5 nodes


class TestGraphHelpers(unittest.TestCase):
    def test_index_of(self):
        graph = build_mesh(chain(3))
        self.assertEqual(graph.nodes[graph.index_of("N1")].ident, "N1")

    def test_index_of_raises_for_unknown_ident(self):
        with self.assertRaises(KeyError):
            build_mesh(chain(3)).index_of("NOPE")

    def test_nearest_node(self):
        graph = build_mesh(chain(5))
        # Closest to N2, which sits at 41.0N.
        self.assertEqual(graph.nodes[nearest_node(graph, 41.02, -90.0)].ident, "N2")


if __name__ == "__main__":
    unittest.main()
