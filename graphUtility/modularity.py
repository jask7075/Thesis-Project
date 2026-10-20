"""Modularity matrix and the spectral stopping criterion, shared by quantum and classical solvers."""

import networkx as nx
import numpy as np


def total_edge_weight(G):
    """Returns m, the total edge weight of G (the edge count for an unweighted graph).

    Edges without a 'weight' attribute count as 1, matching nx.to_numpy_array,
    so m is always consistent with the modularity matrix built below.
    """
    return G.size(weight="weight")


def create_modularity_matrix(G):
    """Calculates the modularity matrix B for a given graph G.

        B_ij = A_ij - (k_i * k_j) / (2m)

    A is the weighted adjacency matrix and k the weighted degree vector, so
    the null-model term is consistent with A on weighted graphs (e.g. Les
    Misérables). Row/column i corresponds to the i-th node of list(G.nodes()),
    which is the same order every encoder and decoder uses.

    The modularity matrix quantifies how much denser connections within
    communities are compared to the expectation in a random graph with the
    same degree distribution.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        numpy.ndarray: The NxN modularity matrix B.
    """
    A = nx.to_numpy_array(G, nodelist=list(G.nodes()))
    degrees = A.sum(axis=1)
    two_m = degrees.sum()

    if two_m == 0:  # Avoid division by zero for graphs with no edges
        modularity_matrix = A.copy()
    else:
        modularity_matrix = A - np.outer(degrees, degrees) / two_m

    print("\n\nModularity Matrix for Graph G:")
    print(modularity_matrix)
    return modularity_matrix


def spectral_stop(B):
    """Checks the spectral stopping criterion for graph partitioning.

    Computes the largest eigenvalue of the modularity matrix B. If it is
    non-positive, no further community structure can be extracted and
    the algorithm should stop.

    Idea described in:
    Newman, M. E. J. (2006). "Modularity and community structure in networks."
    Proceedings of the National Academy of Sciences, 103(23), 8577-8582.
    DOI: 10.1073/pnas.0601602103 | arXiv: physics/0602124

    Args:
        B (numpy.ndarray): The modularity matrix of the (sub)graph.

    Returns:
        bool: True if the maximum eigenvalue <= 0 (stop), False otherwise (continue).
    """
    # Compute all eigenvalues of B; take the real part to discard numerical noise
    lambda_max = np.max(np.linalg.eigvals(B).real)
    print(f"\n\nMaximum eigenvalue of the modularity matrix: {lambda_max}")

    # A non-positive maximum eigenvalue means splitting this (sub)graph
    # would not improve modularity, so signal the algorithm to stop
    return lambda_max <= 0
