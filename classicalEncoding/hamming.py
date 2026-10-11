"""Classical Hamming(7,4) / Hamming(8,4) baseline: simulated annealing over binary strings.

Total objective to MINIMISE — the same energy as the QAOA Hamiltonian
(quantumEncodings.hamming), evaluated on basis states:

  H_total = H_mod + PP

  H_mod = -1/(4m) Σ_q Σ_{i<j} B_ij v_{i,q} v_{j,q}    (modularity)
  PP    = -λ Σ_i Σ_{p∈P} Π_{q∈p} v_{i,q}              (parity penalty)

x_{i,q} ∈ {0, 1}  — strictly binary, no relaxation
v_{i,q} = 2x_{i,q} - 1  maps {0,1} → {-1,+1}

Optimiser: simulated annealing over {0,1}^(N×L) with random bit-flip moves,
           L = codeword length (7 or 8). No rounding needed at decode.
"""

import numpy as np

from graphUtility.modularity import create_modularity_matrix, total_edge_weight
from graphUtility.plotting import display_graph, plot_communities, plot_convergence
from quantumEncodings.hamming import decode_hamming_bitstring


def hamming_objective(x_flat, B, N, lambda_penalty, code, m):
    """Computes the Hamming-encoded total energy H_total = H_mod + PP.

    Uses Σ_{i<j} B_ij v_i v_j = (v^T B v - trace(B)) / 2, valid because v_i² = 1.

    Args:
        x_flat (numpy.ndarray): Flattened binary assignment of shape (N*code.length,).
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        lambda_penalty (float): Penalty coefficient controlling the strength of the parity constraint.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        m (float): The total edge weight of the graph.

    Returns:
        float: The total energy H_total (to be minimised).
    """
    V = 2.0 * x_flat.reshape(N, code.length) - 1.0  # {0,1} → {-1,+1}

    pair_sum = sum((V[:, q] @ B @ V[:, q] - np.trace(B)) / 2 for q in range(code.length))
    h_mod = -pair_sum / (4 * m)
    pp = -lambda_penalty * sum(np.prod(V[i, pc]) for i in range(N) for pc in code.parity_checks)

    return h_mod + pp


def run_hamming_sa(B, N, lambda_penalty, code, m, T_init=5.0, T_min=1e-4, maxiter=100000, seed=42):
    """Minimises the Hamming objective using simulated annealing over binary strings.

    At each step a single random bit is flipped and the move is accepted via
    the Metropolis criterion. Temperature decreases geometrically from T_init
    to T_min over maxiter steps.

    Args:
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        lambda_penalty (float): Penalty coefficient for the parity constraint.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        m (float): The total edge weight of the graph.
        T_init (float): Initial temperature for the annealing schedule.
        T_min (float): Final temperature at which annealing stops.
        maxiter (int): Total number of bit-flip proposals.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            - best_x (numpy.ndarray): The best binary solution found.
            - best_val (float): The lowest objective value achieved.
            - tracker (list[float]): Best-so-far objective value at each improvement.
    """
    rng = np.random.default_rng(seed)
    n_vars = N * code.length
    cooling = (T_min / T_init) ** (1.0 / maxiter)  # Geometric cooling factor applied each iteration

    x = rng.integers(0, 2, size=n_vars).astype(float)
    val = hamming_objective(x, B, N, lambda_penalty, code, m)
    best_x = x.copy()
    best_val = val
    tracker = [val]

    T = T_init
    for _ in range(maxiter):
        # Propose: flip one random bit
        idx = rng.integers(0, n_vars)
        x_new = x.copy()
        x_new[idx] = 1.0 - x_new[idx]

        new_val = hamming_objective(x_new, B, N, lambda_penalty, code, m)
        delta = new_val - val

        # Metropolis: always accept improvements, accept worse moves with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            x = x_new
            val = new_val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                tracker.append(best_val)

        T *= cooling

    print(f"SA finished | best H_total = {best_val:.4f} | improvements = {len(tracker)-1}")
    return best_x, best_val, tracker


def run_hamming_classical(G, N, K, lambda_penalty, code, maxiter=100000, seed=42):
    """Runs the full classical simulated annealing pipeline for a Hamming encoding.

    Constructs the modularity matrix, runs simulated annealing to find the
    optimal binary assignment, decodes the result with syndrome correction,
    and displays the convergence plot and final community partition.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        lambda_penalty (float): Penalty coefficient for the parity constraint.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        maxiter (int): Total number of bit-flip proposals.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: bitstring, assignments, optimal energy and convergence tracker.
    """
    display_graph(G, "Input Graph")
    B = create_modularity_matrix(G)
    m = total_edge_weight(G)

    best_x, best_val, tracker = run_hamming_sa(B, N, lambda_penalty, code, m, maxiter=maxiter, seed=seed)
    bitstring = [int(b) for b in best_x]
    assignments, _ = decode_hamming_bitstring(bitstring, N, K, code, correct=True)

    print(f"\nOptimal H_total     : {best_val:.4f}")
    print(f"Community assignments: {assignments}")

    plot_convergence(tracker, f"Simulated annealing — {code.name} best-so-far",
                     xlabel="Improvement number", ylabel="Best H_total found", step=True)
    plot_communities(G, assignments, f"Detected Communities — {code.name}, K={K}")

    return {"bitstring": bitstring, "assignments": assignments,
            "objective": float(best_val), "value_tracker": tracker}
