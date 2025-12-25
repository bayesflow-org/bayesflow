import re
from copy import deepcopy
from typing import TypeAlias

import networkx as nx

Node: TypeAlias = str


def split_node(graph: nx.DiGraph, node: Node) -> nx.DiGraph:
    """
    Splits a node in a graph into two nodes. This is required to determine
    if an inference network can estimate parameters group-wise or if variables
    have to be estimated jointly.
    """
    subgraph = extract_subgraph(graph, node)
    other_nodes = set(graph.nodes).difference(subgraph.nodes)
    split_graph = nx.DiGraph(graph.subgraph(other_nodes))

    for i in [1, 2]:
        sorted_nodes = list(nx.lexicographical_topological_sort(subgraph))
        renamed_nodes = [add_suffix(n, suffix=i) for n in sorted_nodes]

        # add nodes from subgraph to new graph
        for original, renamed in zip(sorted_nodes, renamed_nodes):
            split_graph.add_node(renamed)

            # add metadata
            split_graph = add_split_by_metadata(subgraph, split_graph, original, renamed, node)
            split_graph = add_previous_names_metadata(subgraph, split_graph, original, renamed)

        # add edges
        for parent in graph.predecessors(node):
            split_graph.add_edge(parent, renamed_nodes[0])

        for start, stop in subgraph.edges:
            split_graph.add_edge(add_suffix(start, i), add_suffix(stop, i))

        # mend broken connections (ancestors of descendants of 'node')
        for start, stop in graph.edges:
            if start in other_nodes and stop in subgraph.nodes:
                split_graph.add_edge(start, add_suffix(stop, i))

    return nx.DiGraph(split_graph)


def extract_subgraph(graph: nx.DiGraph, node: Node) -> nx.DiGraph:
    """
    Extracts a subgraph from a larger graph, consisting of a given node and
    all its downstream nodes.
    """
    included_nodes = set([node])
    included_nodes.update(nx.descendants(graph, node))

    subgraph = graph.subgraph(included_nodes)

    return nx.DiGraph(subgraph)


def has_open_path(graph: nx.DiGraph, x: Node, y: Node, known: list[Node]) -> bool:
    """
    Used by the graph inversion algorithm to determine if the nodes in `x` and `y`
    are conditionally independent given the nodes in `known`.
    """
    all_paths = list(nx.all_simple_paths(graph.to_undirected(), x, y))
    is_blocked = [False for _ in all_paths]

    for i, path in enumerate(all_paths):
        # If a node in the path is known and not a collider, it blocks that path.
        # If a node in the path is a collider and not known, it blocks that path.
        for idx, node in enumerate(path[1:-1]):
            is_collider = graph.has_edge(path[idx], path[idx + 1]) & graph.has_edge(path[idx + 2], path[idx + 1])
            if node in known and not is_collider:
                is_blocked[i] = True
            elif is_collider:
                known_descendant = len(nx.descendants(graph, node) & set(known)) != 0
                if node not in known and not known_descendant:
                    is_blocked[i] = True

    if all(is_blocked):
        return False
    else:
        return True


def add_suffix(string: str, suffix: int):
    """
    Adds a suffix to a string, optionally prepending an underscore if one not
    already exists.

    >>> add_suffix("nodename", 1)
    nodename_1

    >>> add_suffix("nodename_12", 3)
    nodename_123
    """
    if bool(re.search(r"_\d+$", string)):
        return string + str(suffix)
    else:
        return string + "_" + str(suffix)


def add_split_by_metadata(
    from_graph: nx.DiGraph,
    to_graph: nx.DiGraph,
    from_node: Node,
    to_node: Node,
    split_node: Node,
) -> nx.DiGraph:
    """
    Given a node `from_node` in `from_graph`, takes existing split_by metadata and
    add it to `to_node` in `to_graph`. Then, append `split_node` to split_by
    entry in `to_node`.

    This is used internally to add a "split_by" annotation field, which contains a
    reference to which nodes caused another node to be split during graph expansion.
    """
    to_graph = deepcopy(to_graph)

    if "split_by" in from_graph.nodes[from_node]:
        # transfer metadata in from_graph to to_graph
        splits = from_graph.nodes[from_node]["split_by"]
        to_graph.nodes[to_node]["split_by"] = splits

        # add split_node to split_on
        if split_node not in splits:
            to_graph.nodes[to_node]["split_by"].append(split_node)
    else:
        to_graph.nodes[to_node]["split_by"] = [split_node]

    return to_graph


def add_previous_names_metadata(
    from_graph: nx.DiGraph, to_graph: nx.DiGraph, from_node: Node, to_node: Node
) -> nx.DiGraph:
    """
    Given a node `from_node` in `from_graph`, takes existing previous_names metadata and
    adds it to `to_node` in `to_graph`. Then, append `from_node` to previous_names
    entry in `to_node`.

    This is used internally to add a "previous_names" node annotation field, which
    contains a reference to previous node names during graph expansion.
    """
    to_graph = deepcopy(to_graph)

    if "previous_names" in from_graph.nodes[from_node]:
        # transfer metadata on subgraph to graph
        previous_names = from_graph.nodes[from_node]["previous_names"]
        to_graph.nodes[to_node]["previous_names"] = previous_names

        # add name to previous_names
        if from_node not in previous_names:
            to_graph.nodes[to_node]["previous_names"].append(from_node)
    else:
        to_graph.nodes[to_node]["previous_names"] = [from_node]

    return to_graph


def sort_nodes_topologically(graph: nx.DiGraph, nodes: list[Node]):
    """
    Orders a list of `nodes` according to the topology defined in `graph`.
    """
    order = list(nx.lexicographical_topological_sort(graph))
    position = {node: i for i, node in enumerate(order)}

    sorted_nodes = sorted(nodes, key=lambda n: position[n])

    return sorted_nodes


def merge_root_nodes(graph: nx.DiGraph):
    """
    Returns a graph with merged root nodes. This reduces the number of
    required inference networks because root nodes can always be estimated jointly
    by a single, top-level inference network.
    """
    root_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    return merge_nodes(graph, root_nodes)


def merge_nodes(graph: nx.DiGraph, nodes: list[Node]):
    """
    Given an input graph, returns a graph where the nodes given in `nodes` are merged
    into a single node. Used for merging root nodes.
    """
    graph = deepcopy(graph)

    for node in nodes[1::]:
        graph = nx.contracted_nodes(graph, nodes[0], node, copy=False, self_loops=False)

    new_name = ", ".join(map(str, nodes))
    graph = nx.relabel_nodes(graph, {nodes[0]: new_name}, copy=False)

    graph.nodes[new_name].clear()
    graph.nodes[new_name]["split_by"] = []
    graph.nodes[new_name]["previous_names"] = []
    graph.nodes[new_name]["merged_from"] = nodes

    return graph
