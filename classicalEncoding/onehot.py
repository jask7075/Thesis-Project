"""Classical one-hot baseline: continuous relaxation optimised with COBYLA.

Objective (maximisation):
  Σ_j  x_j^T B x_j  +  Σ_i γ_i (Σ_j x_{i,j} - 1)²
γ_i = -lambda_penalty  (negative so one-hot violations reduce the objective)

Variables: X[i, j] = x_{i,j}  (continuous relaxation of binary assignment)
Strategy : continuous relaxation → COBYLA → argmax decode → visualise
"""

import numpy as np
from scipy.optimize import minimize

from graphUtility.modularity import create_modularity_matrix
from graphUtility.plotting import display_graph, plot_communities, plot_convergence


def onehot_objective(x_flat, B, N, K, lambda_penalty, tracker):
    """Computes the negated one-hot objective for the continuous relaxation.

    Evaluates the modularity gain minus a penalty for violating the one-hot
    constraint, negated so that minimisation routines maximise it.

    Args:
        x_flat (numpy.ndarray): Flattened assignment matrix of shape (N*K,).
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        lambda_penalty (float): Penalty coefficient for one-hot constraint violations.
        tracker (list): Accumulates the objective value at each evaluation for convergence plots.

    Returns:
        float: Negated objective value.
    """
    X = x_flat.reshape(N, K)

    modularity = sum(X[:, j] @ B @ X[:, j] for j in range(K))
    penalty = -lambda_penalty * np.sum((X.sum(axis=1) - 1) ** 2)  # rows should sum to 1
    value = modularity + penalty

    tracker.append(value)
    return -value


def run_onehot_cobyla(B, N, K, lambda_penalty, n_restarts=10, seed=42):
    """Optimises the one-hot objective using COBYLA with multiple random restarts.

    Multiple Dirichlet-initialised restarts reduce the risk of converging to
    a poor local optimum. The best result across all restarts is returned.

    Args:
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        lambda_penalty (float): Penalty coefficient for the one-hot constraint.
        n_restarts (int): Number of independent random restarts.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            - best_result (scipy.optimize.OptimizeResult): The best optimisation result found.
            - tracker (list[float]): Objective values recorded across all evaluations.
    """
    rng = np.random.default_rng(seed)
    best_result = None
    tracker = []

    # Non-negativity constraints: x_{i,j} >= 0 for all i, j
    constraints = [
        {"type": "ineq", "fun": lambda x, idx=idx: x[idx]}
        for idx in range(N * K)
    ]

    for _ in range(n_restarts):
        # Dirichlet initialisation: rows sum to 1, all entries >= 0
        x0 = rng.dirichlet(np.ones(K), size=N).flatten()

        result = minimize(
            onehot_objective,
            x0,
            args=(B, N, K, lambda_penalty, tracker),
            method="COBYLA",
            constraints=constraints,
            tol=1e-6,
            options={"maxiter": 5000, "rhobeg": 0.5},
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result, tracker


def decode_onehot_assignments(x_flat, N, K):
    """Decodes the relaxed solution: each node joins the community with the largest value.

    Args:
        x_flat (numpy.ndarray): Flattened assignment matrix of shape (N*K,).
        N (int): The number of nodes in the graph.
        K (int): The number of communities.

    Returns:
        list[int]: Community index per node.
    """
    return [int(c) for c in np.argmax(x_flat.reshape(N, K), axis=1)]


def run_onehot_classical(G, N, K, lambda_penalty, n_restarts=10, seed=42):
    """Runs the full classical COBYLA pipeline for the one-hot community detection problem.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        lambda_penalty (float): Penalty coefficient for the one-hot constraint.
        n_restarts (int): Number of independent random restarts.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: assignments, optimal objective and convergence tracker.
    """
    display_graph(G, "Input Graph")
    B = create_modularity_matrix(G)

    result, tracker = run_onehot_cobyla(B, N, K, lambda_penalty, n_restarts, seed)
    assignments = decode_onehot_assignments(result.x, N, K)

    print(f"Optimal objective : {-result.fun:.4f}")
    print(f"Community assignments: {assignments}")
    print(f"Function evaluations : {result.nfev}")

    plot_convergence(tracker, "Classical One-Hot — objective per evaluation",
                     xlabel="Function evaluation", ylabel="Objective value")
    plot_communities(G, assignments, "Detected Communities — classical one-hot")

    return {"assignments": assignments, "objective": float(-result.fun), "value_tracker": tracker}
