"""Classical binary baseline: continuous relaxation optimised with COBYLA.

Objective (maximise):
  H_C = (1/2m) * Σ_n Σ_{i,j} B_ij * (x_{i,n} - 0.5) * (x_{j,n} - 0.5)

Variables: X[i, n] = x_{i,n}  — n-th bit of node i's community label,
           continuous relaxation x_{i,n} ∈ [0, 1]
Decode   : round X to {0,1}, read each row as a binary integer → community index
Optimiser: scipy COBYLA (no gradients needed, same family as the QAOA optimiser)
"""

import numpy as np
from scipy.optimize import minimize

from graphUtility.modularity import create_modularity_matrix, total_edge_weight
from graphUtility.plotting import display_graph, plot_communities, plot_convergence
from quantumEncodings.binary import bits_for_communities, decode_binary_bitstring


def binary_objective(x_flat, B, N, num_bits, m, tracker):
    """Computes the negated binary-encoded modularity objective for the continuous relaxation.

    Evaluates H_C = (1/2m) * Σ_n v_n^T B v_n, where v_n = X[:, n] - 0.5.

    Args:
        x_flat (numpy.ndarray): Flattened assignment matrix of shape (N*num_bits,).
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        num_bits (int): The number of binary bits used to encode each community label.
        m (float): The total edge weight of the graph.
        tracker (list): Accumulates the objective value at each evaluation for convergence plots.

    Returns:
        float: Negated objective value.
    """
    X = x_flat.reshape(N, num_bits)

    value = 0.0
    for bit in range(num_bits):
        v = X[:, bit] - 0.5  # Maps {0,1} → {-0.5, +0.5}
        value += v @ B @ v
    value /= (2 * m)

    tracker.append(value)
    return -value


def run_binary_cobyla(B, N, K, m, n_restarts=10, seed=42):
    """Optimises the binary objective using COBYLA with multiple random restarts.

    Box constraints keep all variables in [0, 1].

    Args:
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        m (float): The total edge weight of the graph.
        n_restarts (int): Number of independent random restarts.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            - best_result (scipy.optimize.OptimizeResult): The best optimisation result found.
            - tracker (list[float]): Objective values recorded across all evaluations.
            - num_bits (int): The number of bits used to encode each community label.
    """
    num_bits = bits_for_communities(K)
    rng = np.random.default_rng(seed)
    best_result = None
    tracker = []

    # Box constraints: 0 <= x_{i,n} <= 1 for all i, n
    constraints = []
    for idx in range(N * num_bits):
        constraints.append({"type": "ineq", "fun": lambda x, i=idx: x[i]})
        constraints.append({"type": "ineq", "fun": lambda x, i=idx: 1 - x[i]})

    for _ in range(n_restarts):
        x0 = rng.uniform(0, 1, size=N * num_bits)

        result = minimize(
            binary_objective,
            x0,
            args=(B, N, num_bits, m, tracker),
            method="COBYLA",
            constraints=constraints,
            tol=1e-6,
            options={"maxiter": 5000},
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result, tracker, num_bits


def decode_binary_assignments(x_flat, N, K):
    """Rounds the relaxed solution to bits and decodes each node's label, wrapped into [0, K).

    Args:
        x_flat (numpy.ndarray): Flattened assignment matrix of shape (N*num_bits,).
        N (int): The number of nodes in the graph.
        K (int): The number of communities.

    Returns:
        list[int]: Community index per node.
    """
    return decode_binary_bitstring((x_flat >= 0.5).astype(int), N, K)


def run_binary_classical(G, N, K, n_restarts=10, seed=42):
    """Runs the full classical COBYLA pipeline for the binary-encoded community detection problem.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        n_restarts (int): Number of independent random restarts.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: assignments, optimal objective and convergence tracker.
    """
    display_graph(G, title="Initial Graph for Community Detection")
    B = create_modularity_matrix(G)
    m = total_edge_weight(G)

    result, tracker, num_bits_used = run_binary_cobyla(B, N, K, m, n_restarts, seed)
    assignments = decode_binary_assignments(result.x, N, K)

    print(f"num_bits used       : {num_bits_used}")
    print(f"Optimal H_C         : {-result.fun:.4f}")
    print(f"Community assignments: {assignments}")
    print(f"Function evaluations : {result.nfev}")

    plot_convergence(tracker, "Classical COBYLA — binary encoding convergence",
                     xlabel="Function evaluation", ylabel="H_C")
    plot_communities(G, assignments, f"Detected Communities — binary encoding (K={K})")

    return {"assignments": assignments, "objective": float(-result.fun), "value_tracker": tracker}
