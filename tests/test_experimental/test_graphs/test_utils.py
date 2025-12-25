from bayesflow.experimental.graphs.utils import has_open_path
import networkx as nx


def test_has_open_path():
    # A -> B -> C
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C")])

    # path A--B--C is open if we don't condition on B
    assert has_open_path(g, "A", "C", known=[]) is True
    assert has_open_path(g, "A", "C", known=["B"]) is False

    # A <- B -> C
    g = nx.DiGraph()
    g.add_edges_from([("B", "A"), ("B", "C")])

    # path is open unless conditioned on B
    assert has_open_path(g, "A", "C", known=[]) is True
    assert has_open_path(g, "A", "C", known=["B"]) is False

    # A -> B <- C
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("C", "B")])

    # collider is blocked unless conditioned on B or descendant of B
    assert has_open_path(g, "A", "C", known=[]) is False
    assert has_open_path(g, "A", "C", known=["B"]) is True

    g.add_edges_from([("B", "D")])
    assert has_open_path(g, "A", "C", known=[]) is False
    assert has_open_path(g, "A", "C", known=["D"]) is True

    # stays blocked when conditioned on unrelated node
    g.add_edges_from([("C", "E")])
    assert has_open_path(g, "A", "C", known=["E"]) is False

    # Two undirected-simple paths between A and C:
    # Path 1: A -> B -> C (chain) (blocked if B known)
    # Path 2: A -> D <- C (collider) (blocked unless D or descendant known)
    #
    # If we condition on B only, Path 1 blocks but Path 2 is still blocked (collider not conditioned) => False
    # If we condition on B and D, Path 1 blocks but Path 2 opens => True
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),  # chain
            ("A", "D"),
            ("C", "D"),  # collider at D
        ]
    )

    assert has_open_path(g, "A", "C", known=["B"]) is False
    assert has_open_path(g, "A", "C", known=["B", "D"]) is True
