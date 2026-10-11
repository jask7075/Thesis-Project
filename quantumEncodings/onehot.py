"""One-hot encoding: node i in community r  <->  qubit i*K + r is |1>, exactly one per node."""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from graphUtility.graphs import create_supernode
from graphUtility.modularity import create_modularity_matrix, spectral_stop, total_edge_weight
from graphUtility.plotting import display_graph, draw_circuit_mpl, plot_communities, plot_convergence
from quantumUtility.ansatz import (INITIAL_BETA, INITIAL_GAMMA, create_qaoa_ansatz,
                                   create_total_cost_hamiltonian, initial_qaoa_params)
from quantumUtility.mixers import create_xy_mixer_hamiltonian
from quantumUtility.simulator import (run_ansatz_simulator_circuit_prep, run_optimizer_simulator,
                                      sample_output)


def create_cost_modularity_hamiltonian(G, B, N, K):
    """Creates the modularity cost Hamiltonian H_mod for the one-hot encoding.

        H_mod = -1/(4m) sum_{r=1}^{k} sum_{i<j} B_ij (1 - Z_{i,r} - Z_{j,r} + Z_{i,r} Z_{j,r})

    Args:
        G (networkx.Graph): The original graph.
        B (numpy.ndarray): The modularity matrix B for the graph G.
        N (int): The number of nodes in the original graph.
        K (int): The number of communities.

    Returns:
        tuple: A tuple containing:
            - num_qubits (int): The total number of qubits required (N*K).
            - modularity_hamiltonian (SparsePauliOp): The constructed cost Hamiltonian.
    """
    pauli_terms = []

    # m: total edge weight of G (edge count for unweighted graphs)
    m = total_edge_weight(G)

    # Node i, community r  ->  qubit i*K + r
    num_qubits = N * K

    if m == 0:
        print("Graph has no edges. The Modularity Hamiltonian will be trivial (zero).")
    else:
        row_sums_B = np.sum(B, axis=1)
        sum_B_ij_i_lt_j = np.sum(np.triu(B, k=1))

        # 1. Constant Term: - (k / (4m)) * sum_{i<j} B_ij
        coeff_const = (-K / (4 * m)) * sum_B_ij_i_lt_j
        if coeff_const != 0:
            pauli_terms.append(('I' * num_qubits, coeff_const))

        for r in range(K):
            for i in range(N):
                # 2. Linear Terms: (1 / (4m)) * (sum_{j!=i} B_ij) * Z_ir
                coeff_linear = (row_sums_B[i] - B[i, i]) / (4 * m)
                if coeff_linear != 0:
                    pauli_string = ['I'] * num_qubits
                    pauli_string[i * K + r] = 'Z'
                    pauli_terms.append(("".join(reversed(pauli_string)), coeff_linear))

                for j in range(i + 1, N):
                    # 3. Quadratic Terms: - (B_ij / (4m)) * Z_ir Z_jr
                    # Interaction between nodes i and j if both are in community r.
                    coeff_quadratic = -B[i, j] / (4 * m)
                    if coeff_quadratic != 0:
                        pauli_string = ['I'] * num_qubits
                        pauli_string[i * K + r] = 'Z'
                        pauli_string[j * K + r] = 'Z'
                        pauli_terms.append(("".join(reversed(pauli_string)), coeff_quadratic))

    modularity_hamiltonian = SparsePauliOp.from_list(pauli_terms, num_qubits=num_qubits)

    print(f"\n\nModularity Hamiltonian (Hz) constructed for {num_qubits} qubits:")
    print(modularity_hamiltonian)
    print(f"Number of terms in Hamiltonian: {len(pauli_terms)}")

    return num_qubits, modularity_hamiltonian


def create_onehot_penalty_hamiltonian(N, num_qubits, K, lambda_penalty):
    """Creates the One-Hot Penalty Hamiltonian, which enforces the one-hot encoding constraint.

        H_pen = sum_i { (lambda/2) sum_{r<s} Z_{i,r} Z_{i,s} - (lambda (k-2)/2) sum_r Z_{i,r} }

    Exactly one qubit per node should be |1>; this penalises a node assigned
    to zero or several communities.

    Args:
        N (int): The number of nodes in the graph.
        num_qubits (int): Total number of qubits in the system (N*K).
        K (int): The number of communities.
        lambda_penalty (float): Strength of the penalty terms.

    Returns:
        qiskit.quantum_info.SparsePauliOp: The constructed Penalty Hamiltonian.
    """
    pauli_terms = []

    if K < 2:
        print("One-Hot Penalty Hamiltonian requires at least two communities (k >= 2).")
    else:
        for i in range(N):
            # Term 1: (lambda/2) * sum_{r<s} Z_ir Z_is
            # Penalises node i being assigned to two communities r and s simultaneously.
            for r in range(K):
                for s in range(r + 1, K):
                    pauli_string = ['I'] * num_qubits
                    pauli_string[i * K + r] = 'Z'
                    pauli_string[i * K + s] = 'Z'
                    pauli_terms.append(("".join(reversed(pauli_string)), lambda_penalty / 2.0))

            # Term 2: - (lambda * (k-2) / 2) * sum_r Z_ir
            # Together with Term 1 this makes "exactly one qubit is |1>" the minimum.
            for r in range(K):
                pauli_string = ['I'] * num_qubits
                pauli_string[i * K + r] = 'Z'
                pauli_terms.append(("".join(reversed(pauli_string)), -lambda_penalty * (K - 2) / 2.0))

    onehot_penalty_hamiltonian = SparsePauliOp.from_list(pauli_terms, num_qubits=num_qubits)

    print(f"\n\nOne Hot Penalty Hamiltonian constructed for {num_qubits} qubits:")
    print(onehot_penalty_hamiltonian)
    print(f"Number of terms in Penalty Hamiltonian: {len(pauli_terms)}")
    return onehot_penalty_hamiltonian


def decode_onehot_bitstring(bitstring, N, K):
    """Decodes a one-hot bitstring into one community index per node.

    Node i's block is bitstring[i*K:(i+1)*K]; its community is the position
    of the first '1'.  A block with no '1' is reported and marked -1.

    Args:
        bitstring (list[int] | list[str]): Measured bits, index j = qubit j.
        N (int): Number of nodes.
        K (int): Number of communities.

    Returns:
        list[int]: Community index per node (-1 for an invalid block).
    """
    bits = [int(b) for b in bitstring]
    assignments = []
    for node_idx in range(N):
        node_k_bits = bits[node_idx * K:(node_idx + 1) * K]
        community = node_k_bits.index(1) if 1 in node_k_bits else -1
        if community == -1:
            print(f"Warning: Node {node_idx} has an invalid community assignment: {node_k_bits}")
        assignments.append(community)

    community_to_nodes = {r: [i for i, c in enumerate(assignments) if c == r] for r in range(K)}
    print("Node community assignments:", assignments)
    print("Community to nodes mapping:", community_to_nodes)
    return assignments


def run_onehot_qaoa(G, N, K, P, lambda_penalty, initial_gamma=INITIAL_GAMMA, initial_beta=INITIAL_BETA):
    """Runs the full QAOA pipeline for community detection using one-hot encoding.

    Constructs the supernode graph, modularity cost Hamiltonian, one-hot penalty
    Hamiltonian, XY mixer, and the QAOA ansatz. Runs classical optimisation on a
    local AerSimulator, then samples the optimised circuit and decodes the result
    into community assignments.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        P (int): The number of QAOA layers.
        lambda_penalty (float): Penalty coefficient for the one-hot constraint Hamiltonian.
        initial_gamma (float): Starting value for every γ parameter.
        initial_beta (float): Starting value for every β parameter.

    Returns:
        dict or None: bitstring, assignments, optimised params and cost tracker;
            None if the spectral criterion finds no community structure.
    """
    K = min(K, N)  # Clamp K to the number of nodes to keep the problem well-defined

    display_graph(G, "Problem Graph")
    display_graph(create_supernode(G, K), "Supernode Structure")

    modularity_matrix = create_modularity_matrix(G)

    # Only proceed if spectral methods say the graph can be split at all
    if spectral_stop(modularity_matrix):
        print("The graph cannot be partitioned into communities based on spectral methods. Exiting.")
        return None

    num_qubits, cost_hamiltonian = create_cost_modularity_hamiltonian(G, modularity_matrix, N, K)
    penalty_hamiltonian = create_onehot_penalty_hamiltonian(N, num_qubits, K, lambda_penalty)
    total_hamiltonian = create_total_cost_hamiltonian(cost_hamiltonian, penalty_hamiltonian, num_qubits)
    mixer_hamiltonian_xy = create_xy_mixer_hamiltonian(num_qubits, N, K)

    # W-state initialisation + XY mixer keep the search inside the one-hot subspace
    circuit = create_qaoa_ansatz(total_hamiltonian, P, N, K, mixer_hamiltonian=mixer_hamiltonian_xy,
                                 initial_state="W")
    print(f"\n\nQAOA Ansatz: ")
    draw_circuit_mpl(circuit, "QAOA Ansatz Circuit")

    # Optimisation process
    transpiled_qc, backend = run_ansatz_simulator_circuit_prep(circuit, method="statevector")
    initial_params = initial_qaoa_params(transpiled_qc, initial_gamma, initial_beta)
    result, value_tracker = run_optimizer_simulator(transpiled_qc, backend, initial_params, total_hamiltonian)
    plot_convergence(value_tracker, "One-hot QAOA - cost per iteration")

    # Result extraction and interpretation
    optimized_circuit = transpiled_qc.assign_parameters(result.x)
    optimized_circuit.measure_all()  # Add measurements to all qubits for SamplerV2

    print(f"\n\nThe circuit with optimized parameters: ")
    draw_circuit_mpl(optimized_circuit, "Optimized Circuit")

    bitstring = sample_output(optimized_circuit, backend, num_qubits)
    print(f"\n\nThe optimized bitstring for the problem in hand: ")
    print(bitstring)

    print(f"\n\nThe community division w.r.t. the input k = {K}:")
    assignments = decode_onehot_bitstring(bitstring, N, K)
    plot_communities(G, assignments, "Graph Partitioned into Communities (one-hot QAOA)")

    return {"bitstring": bitstring, "assignments": assignments,
            "params": list(result.x), "value_tracker": value_tracker}
