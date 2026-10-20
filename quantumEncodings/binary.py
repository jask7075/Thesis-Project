"""Binary encoding: node i's community label is a ceil(log2 K)-bit integer on qubits i*num_bits + b."""

import math

from qiskit.quantum_info import SparsePauliOp

from graphUtility.graphs import create_supernode
from graphUtility.modularity import create_modularity_matrix, total_edge_weight
from graphUtility.plotting import display_graph, draw_circuit_mpl, plot_communities, plot_convergence
from quantumUtility.ansatz import INITIAL_BETA, INITIAL_GAMMA, create_qaoa_ansatz, initial_qaoa_params
from quantumUtility.simulator import (run_ansatz_simulator_circuit_prep, run_optimizer_simulator,
                                      sample_output)


def bits_for_communities(K):
    """Minimum number of bits needed to label K communities (at least 1)."""
    return math.ceil(math.log2(max(K, 2)))


def create_binary_cost_hamiltonian(G, modularity_matrix, N, num_bits):
    """Constructs the cost Hamiltonian for the binary-encoded community detection problem.

        H = -1/(4m) sum_{i<j} B_ij sum_{b=1}^{num_bits} Z_{i,b} Z_{j,b}

    Each original node i is represented by num_bits qubits (the b-th qubit of
    node i has index i*num_bits + b).

    Args:
        G (networkx.Graph): The original graph.
        modularity_matrix (np.ndarray): NxN matrix B_ij.
        N (int): Number of nodes in the graph.
        num_bits (int): Number of binary bits per node.

    Returns:
        tuple: (num_qubits, cost_hamiltonian) where num_qubits = N * num_bits.
    """
    num_qubits = N * num_bits
    m = total_edge_weight(G)
    pauli_terms = []

    for i in range(N):
        for j in range(i + 1, N):
            coeff = -(1 / (4 * m)) * modularity_matrix[i, j]
            if coeff == 0:
                continue
            # Sum over bits: one ZZ term per bit b (not a product over all bits)
            for b in range(num_bits):
                pauli_str = ['I'] * num_qubits
                pauli_str[i * num_bits + b] = 'Z'
                pauli_str[j * num_bits + b] = 'Z'
                pauli_terms.append(("".join(reversed(pauli_str)), coeff))

    cost_hamiltonian = SparsePauliOp.from_list(pauli_terms, num_qubits=num_qubits)
    return num_qubits, cost_hamiltonian


def decode_binary_bitstring(bitstring, N, K):
    """Decodes a binary-encoded bitstring into one community index per node.

    The bits for node i occupy bitstring[i*num_bits:(i+1)*num_bits] and are
    read as a big-endian integer, wrapped into [0, K).

    Args:
        bitstring (list[int] | list[str]): Bits, length N*num_bits.
        N (int): Number of nodes.
        K (int): Number of communities.

    Returns:
        list[int]: Community index per node.
    """
    num_bits = len(bitstring) // N
    bits = "".join(str(int(b)) for b in bitstring)
    return [int(bits[i * num_bits:(i + 1) * num_bits], 2) % K for i in range(N)]


def run_binary_qaoa(G, N, K, P, initial_gamma=INITIAL_GAMMA, initial_beta=INITIAL_BETA):
    """Runs the full QAOA pipeline for community detection using binary encoding.

    Constructs the supernode graph, cost Hamiltonian, and QAOA ansatz for the
    binary-encoded community detection problem. Runs classical optimisation on
    a local AerSimulator, then samples the optimised circuit and decodes the
    result into community assignments.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The requested number of communities.
        P (int): The number of QAOA layers.
        initial_gamma (float): Starting value for every γ parameter.
        initial_beta (float): Starting value for every β parameter.

    Returns:
        dict: bitstring, assignments, optimised params and cost tracker.
    """
    display_graph(G, title="Initial Graph for Community Detection")

    # The maximum number of communities cannot exceed the total number of nodes
    K = min(K, N)
    print(f"KMAX = {N}")

    num_bits = bits_for_communities(K)

    display_graph(create_supernode(G, num_bits), "Supernode Structure")

    modularity_matrix = create_modularity_matrix(G)
    num_qubits, cost_hamiltonian = create_binary_cost_hamiltonian(G, modularity_matrix, N, num_bits)

    circuit = create_qaoa_ansatz(cost_hamiltonian, P, N, K)
    print(f"\n\nQAOA Ansatz: ")
    draw_circuit_mpl(circuit, "QAOA Ansatz Circuit")

    # Optimisation process
    transpiled_qc, backend = run_ansatz_simulator_circuit_prep(circuit)
    initial_params = initial_qaoa_params(transpiled_qc, initial_gamma, initial_beta)
    result, value_tracker = run_optimizer_simulator(transpiled_qc, backend, initial_params, cost_hamiltonian)
    plot_convergence(value_tracker, "Binary QAOA - cost per iteration")

    # Result extraction and interpretation
    optimized_circuit = transpiled_qc.assign_parameters(result.x)
    print(f"\n\nThe circuit with optimized parameters: ")
    draw_circuit_mpl(optimized_circuit, "Optimized Circuit")

    bitstring = sample_output(optimized_circuit, backend, num_qubits)
    print(f"\n\nThe optimized bitstring for the problem in hand: ")
    print(bitstring)

    print(f"\n\nThe community division w.r.t. the input k = {K}:")
    assignments = decode_binary_bitstring(bitstring, N, K)
    print("Node community assignments:", assignments)
    plot_communities(G, assignments, "Detected communities (binary encoding)")

    return {"bitstring": bitstring, "assignments": assignments,
            "params": list(result.x), "value_tracker": value_tracker}
