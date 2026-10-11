"""Hamming(7,4) / Hamming(8,4) encoding: each node is a codeword block; its 4 data bits carry the community.

Also holds the code definitions and syndrome decoding, which the classical
Hamming baseline reuses.
"""

from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import SamplerV2 as Sampler

from graphUtility.graphs import create_supernode
from graphUtility.modularity import create_modularity_matrix, total_edge_weight
from graphUtility.plotting import display_graph, draw_circuit_mpl, plot_communities, plot_convergence
from quantumUtility.ansatz import (INITIAL_BETA, INITIAL_GAMMA, create_qaoa_ansatz,
                                   create_total_cost_hamiltonian, initial_qaoa_params)
from quantumUtility.hardware import (DEFAULT_ENV_PATH, connect_ibm_service, estimate_circuit_fidelity,
                                     run_spsa, sample_output_hardware, select_ibm_backend,
                                     transpile_for_hardware)
from quantumUtility.mixers import create_generator_based_x_mixer
from quantumUtility.simulator import run_ansatz_simulator_circuit_prep, run_optimizer_simulator

# Standard Hamming(7,4) parity check sets (0-indexed bit positions).
# Each bit b has a unique syndrome equal to the binary representation of (b+1),
# so syndrome (s2,s1,s0) directly gives the error position:
#   P1 covers bits where (b+1) has bit-0 set: positions 0,2,4,6
#   P2 covers bits where (b+1) has bit-1 set: positions 1,2,5,6
#   P3 covers bits where (b+1) has bit-2 set: positions 3,4,5,6
# This ensures all 7 non-zero 3-bit syndromes are reachable, including (1,1,1).
PARITY_CODES_74 = [
    [0, 2, 4, 6],  # P1
    [1, 2, 5, 6],  # P2
    [3, 4, 5, 6],  # P3
]

# Extended Hamming(8,4): Hamming(7,4) parity checks plus an overall parity check.
# The overall parity check (P4) covers all 8 bits and allows distinguishing
# single-bit errors (correctable) from double-bit errors (detectable only).
PARITY_CODES_84 = [
    [0, 2, 4, 6],              # P1
    [1, 2, 5, 6],              # P2
    [3, 4, 5, 6],              # P3
    [0, 1, 2, 3, 4, 5, 6, 7],  # P4: overall parity
]

# Generator matrices (0-indexed support positions of each generator row).
#
# These are the DUAL object to PARITY_CODES_* above: a parity-check row is a
# constraint ("these bits XOR to 0"), whereas a generator row is itself a
# codeword, and the rows span the whole 16-word code.
#
# Row d is obtained by encoding the message that is 1 at data bit d and 0
# elsewhere, i.e. its support is d together with every parity bit whose check
# covers d:
#       g_d = {d} u {p_j : d in P_j},   p_j = 2^j - 1
# Parity bits sit at positions 2^j - 1 (0,1,3 for (7,4); 0,1,3,7 for (8,4)),
# so the data bits are 2,4,5,6 in both codes.  Equivalently, bit b carries
# syndrome bin(b+1), so column b of H reads off directly:
#       d=2 -> 3 = 011 -> P1,P2      d=5 -> 6 = 110 -> P2,P3
#       d=4 -> 5 = 101 -> P1,P3      d=6 -> 7 = 111 -> P1,P2,P3
#
# Use these (not PARITY_CODES_*) as the atoms of create_generator_based_x_mixer:
# X^(x)g_k maps |c> -> |c XOR g_k>, so codeword-valued atoms keep the state in
# the code space, and a spanning set is needed to reach all 16 codewords per
# node.  The 3 rows of PARITY_CODES_74 are codewords but span only 8 of the 16,
# so they explore half the code space; PARITY_CODES_84 is self-dual and does
# span, which is why only the (7,4) case actually differs.
GENERATOR_CODES_74 = [
    [0, 1, 2],     # g2  = 1110000, weight 3
    [0, 3, 4],     # g4  = 1001100, weight 3
    [1, 3, 5],     # g5  = 0101010, weight 3
    [0, 1, 3, 6],  # g6  = 1101001, weight 4
]

# Extended Hamming(8,4): the overall parity bit (position 7) is set on every
# odd-weight (7,4) generator row, making all four rows weight 4.  The resulting
# code is self-dual with weight distribution {0:1, 4:14, 8:1} and d=4.
GENERATOR_CODES_84 = [
    [0, 1, 2, 7],  # g2 = 11100001, weight 4
    [0, 3, 4, 7],  # g4 = 10011001, weight 4
    [1, 3, 5, 7],  # g5 = 01010101, weight 4
    [0, 1, 3, 6],  # g6 = 11010010, weight 4
]


def get_data_bit_indices(codeword_length):
    """Return 0-indexed positions of the 4 data bits in a Hamming(n,4) codeword.

    Parity bits sit at positions where (index+1) is a power of 2, i.e.
    (index+1) & index == 0.  All other positions are data bits.

        Hamming(7,4): parity @ {0,1,3}   → data @ [2, 4, 5, 6]
        Hamming(8,4): parity @ {0,1,3,7} → data @ [2, 4, 5, 6]

    Args:
        codeword_length (int): Codeword length (7 or 8).

    Returns:
        list of int: Sorted data-bit positions.
    """
    return [i for i in range(codeword_length) if (i + 1) & i != 0]


@dataclass(frozen=True)
class HammingCode:
    """Everything that differs between Hamming(7,4) and Hamming(8,4).

    Attributes:
        name (str): Display name, e.g. "Hamming(7,4)".
        length (int): Codeword length = qubits per node (7 or 8).
        parity_checks (list[list[int]]): Parity check sets (syndrome decoding, penalty, initial state).
        generator_rows (list[list[int]]): Generator rows (code-preserving mixer atoms).
        default_mixer (str): Mixer used when none is requested: "generator" or "x".
    """
    name: str
    length: int
    parity_checks: list
    generator_rows: list
    default_mixer: str

    @property
    def data_bits(self):
        return get_data_bit_indices(self.length)


# Hamming(8,4) keeps the plain X mixer with the default |+>^n start state.
HAMMING_74 = HammingCode("Hamming(7,4)", 7, PARITY_CODES_74, GENERATOR_CODES_74, default_mixer="generator")
HAMMING_84 = HammingCode("Hamming(8,4)", 8, PARITY_CODES_84, GENERATOR_CODES_84, default_mixer="x")
HAMMING_CODES = {"hamming74": HAMMING_74, "hamming84": HAMMING_84}


def create_modularity_penalty_hamiltonian(G, N, B, codeword_length):
    """Creates the modularity Hamiltonian for the Hamming(7,4) and Hamming(8,4) encodings.

    Implements, with the same 1/(4m) normalisation and i<j convention as the
    one-hot and binary encodings so energies are comparable across encodings:

        H_mod = -1/(4m) sum_{q=0}^{L-1} sum_{i<j} B_ij Z_{i,q} Z_{j,q}

    Diagonal pairs (i == j) are excluded: Z_{i,q} Z_{i,q} = I, so they only
    shift the energy by a constant.

    Qubit mapping: node i, bit q  ->  i * codeword_length + q

    Args:
        G (networkx.Graph): The graph representing the problem.
        N (int): Number of nodes in the graph.
        B (numpy.ndarray): Modularity matrix of shape (N, N).
        codeword_length (int): Qubits per node / codeword length (7 or 8).

    Returns:
        tuple:
            - num_qubits (int): Total number of qubits (N * codeword_length).
            - mp_hamiltonian (SparsePauliOp): The modularity Hamiltonian.
    """
    num_qubits = N * codeword_length

    m = total_edge_weight(G)
    if m == 0:
        print("Graph has no edges. Modularity Penalty Hamiltonian is trivially zero.")
        return num_qubits, SparsePauliOp.from_list([], num_qubits=num_qubits)

    pauli_terms = []

    for i in range(N):
        for j in range(i + 1, N):
            coeff = -B[i, j] / (4 * m)
            if abs(coeff) < 1e-12:
                continue

            for q in range(codeword_length):
                pauli_str = ['I'] * num_qubits
                pauli_str[i * codeword_length + q] = 'Z'
                pauli_str[j * codeword_length + q] = 'Z'
                pauli_terms.append((''.join(reversed(pauli_str)), coeff))

    mp_hamiltonian = SparsePauliOp.from_list(pauli_terms, num_qubits=num_qubits).simplify()

    print(f"\nModularity Penalty Hamiltonian (Hamming {codeword_length},4) constructed for {num_qubits} qubits.")
    print(f"Number of terms after simplification: {len(mp_hamiltonian)}")

    return num_qubits, mp_hamiltonian


def create_parity_penalty_hamiltonian(N, codeword_length, parity_checks, lambda_penalty=1.0):
    """Creates the Parity Penalty Hamiltonian for the Hamming encodings.

    Implements:
        PP = -lambda * sum_{i in V} sum_{P in P} prod_{q in P} Z_{i,q}

    Every valid codeword satisfies all checks (each product is +1), so valid
    codewords sit lambda * |P| lower than strings violating a check.

    Why lambda_penalty?
    -------------------
    Without scaling, the parity penalty and the modularity term may have
    very different energy scales, causing the optimiser to ignore whichever
    is smaller.  lambda_penalty sets the constraint strength so it dominates
    over invalid (non-codeword) states while still leaving the modularity
    term visible.  Note the modularity term is normalised by 1/(4m), so
    lambda should be of the order of max|B_ij| * L / (4m), not of B itself.

    Qubit mapping: node i, bit q  ->  i * codeword_length + q

    Args:
        N (int): Number of nodes in the graph.
        codeword_length (int): Qubits per node / codeword length (7 or 8).
        parity_checks (list(list(int))): Parity check sets for Hamming(7,4) or Hamming(8,4).
        lambda_penalty (float): Penalty strength multiplier. Default 1.0.

    Returns:
        tuple:
            - num_qubits (int): Total number of qubits (N * codeword_length).
            - pp_hamiltonian (SparsePauliOp): The Parity Penalty Hamiltonian.
    """
    num_qubits = N * codeword_length

    pauli_terms = []
    for i in range(N):
        for parity_set in parity_checks:
            # Multi-qubit Z term: -lambda * Z_{i,q1} Z_{i,q2} ... Z_{i,q|P|}
            pauli_str = ['I'] * num_qubits
            for q in parity_set:
                pauli_str[i * codeword_length + q] = 'Z'
            pauli_terms.append((''.join(reversed(pauli_str)), -lambda_penalty))

    pp_hamiltonian = SparsePauliOp.from_list(pauli_terms, num_qubits=num_qubits).simplify()

    print(f"\nParity Penalty Hamiltonian (Hamming {codeword_length},4) constructed for {num_qubits} qubits.")
    print(f"lambda_penalty = {lambda_penalty}")
    print(f"Number of terms after simplification: {len(pp_hamiltonian)}")

    return num_qubits, pp_hamiltonian


def build_syndrome_table(parity_checks, codeword_length):
    """Map every single-bit error pattern to the bit position it corrupts.

    For each bit index b in 0..codeword_length-1, the syndrome produced by a
    single flip at b is the tuple of parities that include b.  Storing the
    inverse gives an O(1) lookup during decoding.

    Args:
        parity_checks (list of list of int): Parity check index sets.
        codeword_length (int): Number of bits per codeword (7 or 8).

    Returns:
        dict: syndrome tuple -> bit index of the single-bit error.
    """
    syndrome_to_bit = {}
    for bit in range(codeword_length):
        syndrome = tuple(1 if bit in parity_set else 0 for parity_set in parity_checks)
        syndrome_to_bit[syndrome] = bit
    return syndrome_to_bit


def compute_syndrome(codeword, parity_checks):
    """Compute the syndrome of a codeword.

    Each entry of the returned tuple is the XOR of the codeword bits
    indexed by the corresponding parity check set.

    Args:
        codeword (list of str | list of int): The codeword bits.
        parity_checks (list of list of int): Parity check index sets.

    Returns:
        tuple of int: Syndrome bits (0 or 1 per check).
    """
    return tuple(
        sum(int(codeword[b]) for b in parity_set) % 2
        for parity_set in parity_checks
    )


def correct_codeword(codeword, parity_checks, syndrome_table, codeword_length):
    """Detect and (if possible) correct a single-bit error in one codeword.

    Hamming(7,4):  corrects all single-bit errors.
    Hamming(8,4):  uses the overall parity bit (last check) to distinguish
                   single-bit errors (correctable) from double-bit errors
                   (detectable only).

    Args:
        codeword (list of str): '0'/'1' characters of length codeword_length.
        parity_checks (list of list of int): Parity check index sets.
        syndrome_table (dict): Output of build_syndrome_table().
        codeword_length (int): 7 or 8.

    Returns:
        tuple:
            - corrected (list of str): Codeword after correction (unchanged if no error).
            - error_bit (int or None): Index of the flipped bit, or None.
            - status (str): Human-readable description.
    """
    syndrome = compute_syndrome(codeword, parity_checks)
    corrected = list(codeword)

    if all(s == 0 for s in syndrome):
        return corrected, None, "no error"

    if codeword_length == 8:
        inner_syndrome = syndrome[:-1]
        overall_parity = syndrome[-1]

        if all(s == 0 for s in inner_syndrome) and overall_parity == 1:
            # Error is in the overall parity bit itself
            error_bit = codeword_length - 1
            corrected[error_bit] = '0' if corrected[error_bit] == '1' else '1'
            return corrected, error_bit, "single-bit error corrected (overall parity bit)"

        if any(s != 0 for s in inner_syndrome) and overall_parity == 1:
            # Single-bit error: locate and correct
            if syndrome in syndrome_table:
                error_bit = syndrome_table[syndrome]
                corrected[error_bit] = '0' if corrected[error_bit] == '1' else '1'
                return corrected, error_bit, f"single-bit error corrected (bit {error_bit})"

        if any(s != 0 for s in inner_syndrome) and overall_parity == 0:
            return corrected, None, "double-bit error detected (uncorrectable)"

    else:
        # Hamming(7,4): every non-zero syndrome maps to exactly one bit
        if syndrome in syndrome_table:
            error_bit = syndrome_table[syndrome]
            corrected[error_bit] = '0' if corrected[error_bit] == '1' else '1'
            return corrected, error_bit, f"single-bit error corrected (bit {error_bit})"

    return corrected, None, f"unknown syndrome {syndrome}"


def correct_hamming_errors(bitstring, N, codeword_length, parity_checks):
    """Detect and correct Hamming code errors across the full measurement bitstring.

    Each node occupies a contiguous block of codeword_length bits.  Syndrome
    decoding is applied independently to every node's codeword.

    Args:
        bitstring (list of str | list of int | str): Bits, length N * codeword_length.
        N (int): Number of graph nodes.
        codeword_length (int): Bits per node (7 or 8).
        parity_checks (list of list of int): Parity check index sets (PARITY_CODES_74 or _84).

    Returns:
        list of str: Corrected bitstring ('0'/'1') of the same length.
    """
    bits = [str(int(b)) for b in bitstring]
    syndrome_table = build_syndrome_table(parity_checks, codeword_length)
    corrected_answer = list(bits)

    print(f"\nHamming({codeword_length},4) Syndrome Decoding")
    print(f"{'Node':<6} {'Codeword':<{codeword_length+2}} {'Syndrome':<{len(parity_checks)+2}} {'Status':<42} Corrected")
    print("-" * (codeword_length * 2 + len(parity_checks) + 56))

    errors_found = 0
    for i in range(N):
        start = i * codeword_length
        codeword = bits[start:start + codeword_length]
        syndrome = compute_syndrome(codeword, parity_checks)
        corrected, error_bit, status = correct_codeword(codeword, parity_checks, syndrome_table,
                                                        codeword_length)
        corrected_answer[start:start + codeword_length] = corrected

        changed = "  *" if corrected != codeword else ""
        if error_bit is not None:
            errors_found += 1
        print(
            f"{i:<6} {''.join(codeword):<{codeword_length+2}} "
            f"{''.join(map(str, syndrome)):<{len(parity_checks)+2}} "
            f"{status:<42} {''.join(corrected)}{changed}"
        )

    print()
    if errors_found == 0:
        print("All codewords valid — no corrections needed.")
    else:
        print(f"{errors_found} node(s) corrected.")
        print(f"Corrected bitstring: {''.join(corrected_answer)}")

    return corrected_answer


def assignments_from_data_bits(bitstring, N, K, codeword_length):
    """Community per node from the 4 data bits of each codeword (big-endian) modulo K.

    Args:
        bitstring (list of str | list of int): Bits, length N * codeword_length.
        N (int): Number of nodes.
        K (int): Number of communities.
        codeword_length (int): 7 or 8.

    Returns:
        list[int]: Community index per node.
    """
    data_indices = get_data_bit_indices(codeword_length)
    bits = [str(int(b)) for b in bitstring]
    return [int(''.join(bits[i * codeword_length + q] for q in data_indices), 2) % K
            for i in range(N)]


def decode_hamming_bitstring(bitstring, N, K, code, correct=True):
    """Optionally syndrome-correct a bitstring, then read communities from the data bits only.

    Args:
        bitstring (list of str | list of int | str): Bits, length N * code.length, index j = qubit j.
        N (int): Number of nodes.
        K (int): Number of communities.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        correct (bool): Apply syndrome correction first.

    Returns:
        tuple:
            - assignments (list[int]): Community index per node.
            - bits (list[str]): The (corrected) bitstring that was decoded.
    """
    bits = [str(int(b)) for b in bitstring]
    if correct:
        bits = correct_hamming_errors(bits, N, code.length, code.parity_checks)

    data_indices = code.data_bits
    assignments = assignments_from_data_bits(bits, N, K, code.length)

    print(f"\nCommunity assignment from data bits {data_indices} (K={K}):")
    print(f"{'Node':<6} {'Codeword':<{code.length + 2}} {'Data bits':<10} {'Value':<7} Community")
    print("-" * (code.length + 32))
    for i in range(N):
        codeword = bits[i * code.length:(i + 1) * code.length]
        data_bits = ''.join(codeword[q] for q in data_indices)
        print(f"{i:<6} {''.join(codeword):<{code.length + 2}} {data_bits:<10} "
              f"{int(data_bits, 2):<7} {assignments[i]}")

    return assignments, bits


def sample_hamming_output(G, optimized_circuit, backend, num_qubits, K, code, shots=10000):
    """Sample the optimised QAOA circuit, syndrome-correct each codeword,
    and assign communities from the 4 data bits only (parity bits excluded).

    Pipeline
    --------
    1. Transpile and sample the circuit (Qiskit Sampler).
    2. Take the most probable bitstring.
    3. Apply Hamming syndrome correction.
    4. Read each node's data bits as a binary number → community = value % K.
    5. Colour and display the graph.

    Bit ordering (Qiskit little-endian)
    ------------------------------------
    get_int_counts() returns integers where qubit 0 is the LSB.
    np.binary_repr gives MSB-first strings, so we reverse to get
    index j ↔ qubit j.

    Known issue (left as-is for now): the circuit coming from
    run_ansatz_simulator_circuit_prep has its measurements stripped and
    nothing re-adds them here, so the sampler result has no 'meas' register
    and step 2 raises AttributeError.

    Args:
        G (networkx.Graph): The problem graph.
        optimized_circuit (QuantumCircuit): Circuit with optimal parameters bound.
        backend: AerSimulator or IBM backend.
        num_qubits (int): Total qubits = N * code.length.
        K (int): Number of communities.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        shots (int): Number of measurement shots. Default 10 000.

    Returns:
        tuple: (assignments (list[int]), corrected bitstring (list[str])).
    """
    # ── 1. Re-transpile ───────────────────────────────────────────────────────
    final_circuit = transpile(optimized_circuit, backend)

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    job = sampler.run([final_circuit], shots=shots)
    counts_int = job.result()[0].data.meas.get_int_counts()

    total = sum(counts_int.values())
    dist_int = {k: v / total for k, v in counts_int.items()}

    most_likely_int = max(dist_int, key=dist_int.get)
    print(f"Most probable outcome : integer {most_likely_int}  "
          f"(probability {dist_int[most_likely_int]:.4f})")

    bitstring = list(reversed(np.binary_repr(most_likely_int, width=num_qubits)))  # index j = qubit j
    print(f"Raw bitstring (qubit-0 first): {''.join(bitstring)}")

    # ── 3-4. Syndrome correction, then decode communities from data bits only ─
    N = G.number_of_nodes()
    assignments, corrected = decode_hamming_bitstring(bitstring, N, K, code, correct=True)

    # ── 5. Colour the graph ───────────────────────────────────────────────────
    plot_communities(G, assignments, f"Detected communities — {code.name}, data bits only")

    return assignments, corrected


def build_hamming_qaoa_circuit(G, N, K, P, lambda_penalty, code, mixer=None):
    """Builds the Hamming QAOA problem: Hamiltonians and logical ansatz.

    Shared by the simulator and hardware pipelines, and by the analysis
    script when it rebuilds a saved hardware run.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        P (int): The number of QAOA layers.
        lambda_penalty (float): Penalty coefficient for the parity constraint Hamiltonian.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        mixer (str): "generator" (code-preserving generator X mixer + Steane |+_L>
            start) or "x" (default X mixer + |+>^n start). None uses code.default_mixer.

    Returns:
        tuple: (circuit (QuantumCircuit), total_hamiltonian (SparsePauliOp), num_qubits (int)).
    """
    mixer = mixer or code.default_mixer

    display_graph(create_supernode(G, code.length), "Supernode Structure")

    modularity_matrix = create_modularity_matrix(G)
    num_qubits, modularity_hamiltonian = create_modularity_penalty_hamiltonian(G, N, modularity_matrix,
                                                                               code.length)
    _, penalty_hamiltonian = create_parity_penalty_hamiltonian(N, code.length, code.parity_checks,
                                                               lambda_penalty)
    total_hamiltonian = create_total_cost_hamiltonian(modularity_hamiltonian, penalty_hamiltonian, num_qubits)

    if mixer == "generator":
        # Mixer built from generator rows, so every atom is a codeword and the
        # code space is preserved; the Steane start puts every node in it.
        mixer_hamiltonian = create_generator_based_x_mixer(num_qubits, N, code.length, code.generator_rows)
        circuit = create_qaoa_ansatz(total_hamiltonian, P, N, K, mixer_hamiltonian=mixer_hamiltonian,
                                     initial_state="steane", parity_checks=code.parity_checks)
    elif mixer == "x":
        circuit = create_qaoa_ansatz(total_hamiltonian, P, N, K)
    else:
        raise ValueError(f"Unknown mixer {mixer!r}; expected 'generator' or 'x'.")

    return circuit, total_hamiltonian, num_qubits


def run_hamming_qaoa(G, N, K, P, lambda_penalty, code, mixer=None,
                     initial_gamma=INITIAL_GAMMA, initial_beta=INITIAL_BETA):
    """Runs the full QAOA pipeline for community detection using a Hamming encoding.

    Constructs the supernode graph, modularity and parity penalty Hamiltonians,
    and the QAOA ansatz. Runs classical optimisation on a local AerSimulator,
    then samples the optimised circuit and decodes the result using syndrome
    correction to obtain final community assignments.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        P (int): The number of QAOA layers.
        lambda_penalty (float): Penalty coefficient for the parity constraint Hamiltonian.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        mixer (str): "generator" or "x"; None uses code.default_mixer.
        initial_gamma (float): Starting value for every γ parameter.
        initial_beta (float): Starting value for every β parameter.

    Returns:
        dict: bitstring, assignments, optimised params and cost tracker.
    """
    display_graph(G, title="Initial Graph for Community Detection")

    circuit, total_hamiltonian, num_qubits = build_hamming_qaoa_circuit(G, N, K, P, lambda_penalty,
                                                                        code, mixer)
    print(f"\n\nQAOA Ansatz: ")
    draw_circuit_mpl(circuit, "QAOA Ansatz Circuit")

    # "mps" is required, not just faster: the default "automatic"/statevector
    # AerSimulator advertises a memory-derived qubit limit (~29 here), so
    # transpiling a 42-qubit circuit against it fails with
    # KeyError: '(41,) not in target.'  The MPS target covers 63 qubits.
    transpiled_qc, backend = run_ansatz_simulator_circuit_prep(circuit, "mps")
    initial_params = initial_qaoa_params(transpiled_qc, initial_gamma, initial_beta)
    result, value_tracker = run_optimizer_simulator(transpiled_qc, backend, initial_params, total_hamiltonian)
    plot_convergence(value_tracker, f"{code.name} QAOA - cost per iteration")

    optimized_circuit = transpiled_qc.assign_parameters(result.x)
    print(f"\n\nThe circuit with optimized parameters: ")
    draw_circuit_mpl(optimized_circuit, "Optimized Circuit")

    assignments, bitstring = sample_hamming_output(G, optimized_circuit, backend, num_qubits, K, code)
    return {"bitstring": bitstring, "assignments": assignments,
            "params": list(result.x), "value_tracker": value_tracker}


def run_hamming_hardware(G, N, K, P, lambda_penalty, code, mixer=None,
                         n_iterations=20, shots=4096, env_path=DEFAULT_ENV_PATH, seed=123,
                         dry_run=True, initial_gamma=INITIAL_GAMMA, initial_beta=INITIAL_BETA):
    """Runs the full QAOA pipeline for community detection on IBM Quantum HARDWARE.

    The hardware twin of run_hamming_qaoa: identical problem construction, but
    the optimisation and sampling run on a real QPU instead of AerSimulator.

    What differs from the simulator routine
    ---------------------------------------
    Optimiser   SPSA, one job per step regardless of parameter count, because
                both probes are submitted as a single Estimator pub.  On plans
                without Sessions every job queues separately, so job count is
                the binding cost.

    Transpiling Done once, against the real device at optimization_level=3, and
                the Hamiltonian is mapped through the resulting layout with
                apply_layout.

    Sampling    sample_output_hardware(), which transpiles the logical circuit
                with its measurements attached so clbit j is logical qubit j.

    Decoding    Communities are read from the RAW bitstring's data bits.
                Syndrome correction is a separate, inspectable step
                (scripts/analysis.py correct), so its effect can be measured.

    Cost
    ----
    n_iterations jobs for the optimisation, plus 1 for the final sample.
    dry_run=True (the default) stops after transpilation and reports depth,
    two-qubit gate count and an estimated fidelity, submitting nothing.

    Args:
        G (networkx.Graph): The problem graph.
        N (int): The number of nodes in the graph.
        K (int): The number of communities.
        P (int): The number of QAOA layers.
        lambda_penalty (float): Penalty coefficient for the parity constraint Hamiltonian.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        mixer (str): "generator" or "x"; None uses code.default_mixer.
        n_iterations (int): SPSA steps = hardware jobs. Default 20.
        shots (int): Shots per expectation value. Default 4096.
        env_path (str): Path to the .env holding IBM_KEY and CRN.
        seed (int): Seed for the SPSA perturbation directions, for reproducibility.
        dry_run (bool): If True, stop before submitting any job. Default True.
        initial_gamma (float): Starting value for every γ parameter.
        initial_beta (float): Starting value for every β parameter.

    Returns:
        dict: backend name and estimated fidelity; on a real run also the raw
            bitstring, its probability, assignments, optimised params and cost
            tracker -- everything scripts/analysis.py needs to re-sample later.
    """
    display_graph(G, title="Initial Graph for Community Detection")

    # The LOGICAL circuit is kept for sampling later
    circuit, total_hamiltonian, num_qubits = build_hamming_qaoa_circuit(G, N, K, P, lambda_penalty,
                                                                        code, mixer)
    print(f"\n\nQAOA Ansatz: ")
    draw_circuit_mpl(circuit, "QAOA Ansatz Circuit (logical)")

    # ── Connect and pick a device ─────────────────────────────────────────────
    service = connect_ibm_service(env_path)
    ibm_backend = select_ibm_backend(service, circuit.num_qubits)

    # ── Transpile ONCE, Estimator path (measurements stripped) ────────────────
    isa_ansatz = transpile_for_hardware(circuit, ibm_backend, strip_measurements=True)
    isa_hamiltonian = total_hamiltonian.apply_layout(isa_ansatz.layout)

    fidelity = estimate_circuit_fidelity(isa_ansatz, ibm_backend)
    record = {"backend": ibm_backend.name, "estimated_fidelity": fidelity}

    print(f"\nJob budget: {n_iterations} SPSA jobs + 1 sampling job = {n_iterations + 1} jobs "
          f"({shots} shots each)")

    if dry_run:
        print("\n*** DRY RUN — nothing submitted. Pass dry_run=False to run on hardware. ***")
        return record

    if fidelity < 1e-3:
        # Not fatal, just honest: the optimiser will be following noise.
        print(f"\n!!! WARNING: estimated fidelity {fidelity:.2e} — results will be "
              f"dominated by noise. Submitting anyway because dry_run=False. !!!")

    # ── Optimisation: SPSA, one job per iteration ─────────────────────────────
    initial_params = initial_qaoa_params(isa_ansatz, initial_gamma, initial_beta)
    params, value_tracker = run_spsa(isa_ansatz, isa_hamiltonian, ibm_backend, initial_params,
                                     n_iterations=n_iterations, shots=shots, seed=seed)
    plot_convergence(value_tracker, f"{code.name} SPSA on {ibm_backend.name} - cost per iteration")

    # ── Sample the optimised LOGICAL circuit ──────────────────────────────────
    raw_bits, modal_probability = sample_output_hardware(circuit, ibm_backend, num_qubits,
                                                         optimal_params=params, shots=shots)

    assignments, _ = decode_hamming_bitstring(raw_bits, N, K, code, correct=False)
    plot_communities(G, assignments, f"Detected communities — {code.name} on {ibm_backend.name}")

    record.update({"raw_bitstring": [int(b) for b in raw_bits],
                   "modal_probability": modal_probability,
                   "assignments": assignments,
                   "params": [float(p) for p in params],
                   "value_tracker": value_tracker})
    return record
