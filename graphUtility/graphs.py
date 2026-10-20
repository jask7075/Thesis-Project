"""Graph construction: synthetic test graphs, NetworkX benchmarks and the supernode view."""

import networkx as nx


def create_initial_graph(n_nodes):
    """Creates the basic graph G: two 3-node cliques joined by a single bridge.

    Args:
        n_nodes (int): The number of nodes in the graph.

    Returns:
        tuple: A tuple containing:
            - G (networkx.Graph): The initialized graph G.
            - N (int): The number of nodes in the graph.
    """
    G = nx.Graph()
    # Add nodes to the graph from 0 to n_nodes-1
    G.add_nodes_from(range(n_nodes))
    edge_list = []

    # This loop structure creates two cliques of 3 nodes each
    # For i=0, it adds edges for nodes (0,1), (0,2), (1,2) forming a clique of (0,1,2).
    # For i=1, it adds edges for nodes (3,4), (3,5), (4,5) forming a clique of (3,4,5).
    for i in range(2):
        for j in range(3):
            for l in range(j + 1, 3):
                edge_list.append((i * 3 + j, i * 3 + l, {'weight': 1.0}))

    # Add an additional edge connecting the two cliques to form a single graph
    edge_list.append((1, 4, {'weight': 1.0}))
    G.add_edges_from(edge_list)
    print(f"Initial Basic Graph G created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, G.number_of_nodes()


def create_clique_chain_graph(x, y):
    """Creates a graph consisting of y complete graphs (cliques), each with x nodes,
    connected in a cycle — each clique is joined to the next by a single edge,
    and the last clique wraps back to the first.

    Node numbering is contiguous: clique k contains nodes [k*x, ..., k*x + x - 1].
    The bridge between clique k and clique (k+1) % y connects the last node of
    clique k to the first node of clique (k+1) % y.

    Args:
        x (int): Number of nodes in each complete graph (clique size).
        y (int): Number of complete graphs (cliques).

    Returns:
        tuple: A tuple containing:
            - G (networkx.Graph): The constructed graph.
            - N (int): The number of nodes in the graph.

    Example:
        x=3, y=3  ->  three triangles (0-1-2), (3-4-5), (6-7-8)
                       bridged by edges (2,3), (5,6), (8,0)
    """
    G = nx.Graph()
    G.add_nodes_from(range(x * y))

    for k in range(y):
        base = k * x

        # Add all edges within clique k (complete subgraph)
        for i in range(x):
            for j in range(i + 1, x):
                G.add_edge(base + i, base + j, weight=1.0)

        # Bridge: last node of clique k -> first node of clique (k+1) % y
        next_base = ((k + 1) % y) * x
        G.add_edge(base + x - 1, next_base, weight=1.0)

    print(f"Graph created: {y} cliques of size {x} in a cycle  |  "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, G.number_of_nodes()


def get_les_miserables_graph():
    """Returns the Les Misérables co-appearance graph.

    A weighted undirected graph in which nodes represent characters
    from the novel and edges represent co-appearances in the same
    chapter. Commonly used as a benchmark for community detection.

    Returns:
        tuple: (G, N) — the graph and its number of nodes.
    """
    G = nx.les_miserables_graph()
    return G, G.number_of_nodes()


def get_karate_club_graph():
    """Returns Zachary's Karate Club graph.

    A well-known social network graph representing friendships between
    34 members of a karate club. Commonly used as a benchmark for
    community detection, as the club split into two factions.

    Returns:
        tuple: (G, N) — the graph and its number of nodes.
    """
    G = nx.karate_club_graph()
    return G, G.number_of_nodes()


def get_davis_southern_women_graph():
    """Returns the Davis Southern Women bipartite graph.

    A bipartite graph representing the attendance of 18 women at
    14 social events in the American South during the 1930s.
    Commonly used as a benchmark for bipartite community detection.

    Returns:
        tuple: (G, N) — the graph and its number of nodes.
    """
    G = nx.davis_southern_women_graph()
    return G, G.number_of_nodes()


def graph_from_spec(spec):
    """Builds a graph from a short text spec, so scripts and saved results can name it.

    Supported specs:
        "clique-chain:XxY"   y cliques of size x in a cycle, e.g. "clique-chain:3x2"
        "initial"            two 3-node cliques joined by a bridge
        "karate"             Zachary's Karate Club
        "les-miserables"     Les Misérables co-appearance graph
        "davis"              Davis Southern Women graph

    Args:
        spec (str): The graph spec.

    Returns:
        tuple: (G, N) — the graph and its number of nodes.
    """
    name, _, arg = spec.partition(":")
    if name == "clique-chain":
        x, y = (int(v) for v in arg.lower().split("x"))
        return create_clique_chain_graph(x, y)
    if name == "initial":
        return create_initial_graph(6)
    if name == "karate":
        return get_karate_club_graph()
    if name == "les-miserables":
        return get_les_miserables_graph()
    if name == "davis":
        return get_davis_southern_women_graph()
    raise ValueError(f"Unknown graph spec {spec!r}; expected clique-chain:XxY, initial, "
                     f"karate, les-miserables or davis.")


def create_supernode(G, K):
    """Creates the supernode graph from an initial graph G.

    Each node in the original graph G gets K copies (one per qubit of its
    encoding block). The copies of a node form a clique, and an edge (u, v)
    in G becomes an edge between (u, i) and (v, i) for every copy index i.

    Args:
        G (networkx.Graph): The original graph.
        K (int): The number of copies (qubits) per original node.

    Returns:
        networkx.Graph: A new graph with supernode structure, where nodes are
                        labeled as (original_node_id, copy_index).
    """
    G_supernodes = nx.Graph()

    # Copies (u, 0), ..., (u, K-1) of each node form a clique within its supernode
    for u in G.nodes():
        for i in range(K):
            for j in range(i + 1, K):
                G_supernodes.add_edge((u, i), (u, j))

    # If u and v are adjacent in G, connect their matching copy slots
    for u, v in G.edges():
        for i in range(K):
            G_supernodes.add_edge((u, i), (v, i))

    print(f"\n\nSupernode Graph - Number of nodes: {G_supernodes.number_of_nodes()}")
    print(f"Supernode Graph - Number of edges: {G_supernodes.number_of_edges()}")

    return G_supernodes
