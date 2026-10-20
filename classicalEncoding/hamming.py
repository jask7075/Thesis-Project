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
from quantumEncodings.hamming import decode_hamming_bitstring, default_hamming_lambda

# Annealing schedule in units of 1/N, the size of a single-flip modularity change
# (tuned together with HAMMING_LAMBDA_SCALE)
SA_T_INIT_SCALE = 3.0
SA_T_MIN_SCALE = 1e-3


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


def run_hamming_sa(B, N, lambda_penalty, code, m, T_init=None, T_min=None, maxiter=100000, seed=42):
    """Minimises the Hamming objective using simulated annealing over binary strings.

    At each step a single random bit is flipped and the move is accepted via
    the Metropolis criterion. Temperature decreases geometrically from T_init
    to T_min over maxiter steps.

    Each proposal costs O(N) rather than a full O(N^2 L) energy evaluation:
    flipping v_{i,q} only changes the terms that contain it,

        dH_mod = (1/2m) v_{i,q} sum_{j != i} B_ij v_{j,q}
        dPP    = 2 lambda sum_{p contains q} prod_{r in p} v_{i,r}

    Temperature scale
    -----------------
    A single flip changes the modularity term by O(1/N) (the mean degree
    over 2m is exactly 1/N), so the default schedule runs from
    SA_T_INIT_SCALE / N down to SA_T_MIN_SCALE / N: hot enough to cross the
    parity barrier between codewords at the start, cold enough to settle
    into a single partition at the end, for any graph size.

    Args:
        B (numpy.ndarray): The modularity matrix for the graph.
        N (int): The number of nodes in the graph.
        lambda_penalty (float): Penalty coefficient for the parity constraint.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        m (float): The total edge weight of the graph.
        T_init (float): Initial temperature; None uses SA_T_INIT_SCALE / N.
        T_min (float): Final temperature; None uses SA_T_MIN_SCALE / N.
        maxiter (int): Total number of bit-flip proposals.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            - best_x (numpy.ndarray): The best binary solution found.
            - best_val (float): The lowest objective value achieved.
            - tracker (list[float]): Best-so-far objective value at each improvement.
    """
    T_init = SA_T_INIT_SCALE / N if T_init is None else T_init
    T_min = SA_T_MIN_SCALE / N if T_min is None else T_min

    rng = np.random.default_rng(seed)
    L = code.length
    n_vars = N * L
    cooling = (T_min / T_init) ** (1.0 / maxiter)  # Geometric cooling factor applied each iteration
    checks_with_bit = [[pc for pc in code.parity_checks if q in pc] for q in range(L)]
    B_offdiag = B - np.diag(np.diag(B))

    x = rng.integers(0, 2, size=n_vars).astype(float)
    V = 2.0 * x.reshape(N, L) - 1.0  # spins of the current state, kept in sync with x
    val = hamming_objective(x, B, N, lambda_penalty, code, m)
    best_x = x.copy()
    best_val = val
    tracker = [val]

    T = T_init
    for _ in range(maxiter):
        # Propose: flip one random bit (node i, codeword position q)
        idx = rng.integers(0, n_vars)
        i, q = divmod(int(idx), L)
        v = V[i, q]

        delta = v * (B_offdiag[i] @ V[:, q]) / (2 * m)
        delta += 2 * lambda_penalty * sum(np.prod(V[i, pc]) for pc in checks_with_bit[q])

        # Metropolis: always accept improvements, accept worse moves with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            V[i, q] = -v
            x[idx] = 1.0 - x[idx]
            val += delta
            if val < best_val:
                best_val = val
                best_x = x.copy()
                tracker.append(best_val)

        T *= cooling

    print(f"SA finished | best H_total = {best_val:.4f} | improvements = {len(tracker)-1}")
    return best_x, best_val, tracker


def run_hamming_classical(G, N, K, lambda_penalty, code, maxiter=100000, n_restarts=5, seed=42):
    """Runs the full classical simulated annealing pipeline for a Hamming encoding.

    Constructs the modularity matrix, runs simulated annealing n_restarts
    times from different random starts and keeps the lowest energy (single
    runs occasionally settle with two communities merged), decodes the result
    with syndrome correction, and displays the convergence plot and final
    community partition.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        lambda_penalty (float): Penalty coefficient for the parity constraint;
            None uses default_hamming_lambda(N).
        code (HammingCode): HAMMING_74 or HAMMING_84.
        maxiter (int): Bit-flip proposals per annealing run.
        n_restarts (int): Independent annealing runs; the lowest energy is kept.
        seed (int): Random seed for reproducibility (restart r uses seed + r).

    Returns:
        dict: bitstring, assignments, optimal energy and convergence tracker.
    """
    if lambda_penalty is None:
        lambda_penalty = default_hamming_lambda(N)
    print(f"lambda_penalty = {lambda_penalty:.4g}")

    display_graph(G, "Input Graph")
    B = create_modularity_matrix(G)
    m = total_edge_weight(G)

    best_x, best_val, tracker = None, np.inf, []
    for restart in range(n_restarts):
        x, val, run_tracker = run_hamming_sa(B, N, lambda_penalty, code, m, maxiter=maxiter,
                                             seed=seed + restart)
        if val < best_val:
            best_x, best_val, tracker = x, val, run_tracker
    bitstring = [int(b) for b in best_x]
    assignments, _ = decode_hamming_bitstring(bitstring, N, K, code, correct=True)

    print(f"\nOptimal H_total     : {best_val:.4f}")
    print(f"Community assignments: {assignments}")

    plot_convergence(tracker, f"Simulated annealing — {code.name} best-so-far",
                     xlabel="Improvement number", ylabel="Best H_total found", step=True)
    plot_communities(G, assignments, f"Detected Communities — {code.name}, K={K}")

    return {"bitstring": bitstring, "assignments": assignments,
            "objective": float(best_val), "value_tracker": tracker}
