"""Constraint-preserving QAOA mixers: XY (one-hot) and generator-based X (Hamming)."""

from qiskit.quantum_info import SparsePauliOp


def create_xy_mixer_hamiltonian(num_qubits, N, K):
    """Creates the XY mixer Hamiltonian (H_M) for the QAOA algorithm. Only for One-Hot.

    The mixer drives transitions between community assignments of the same
    node while preserving the Hamming weight of each node's block:

        H_M = sum_{i=1}^{n} sum_{r<s} (X_{i,r}X_{i,s} + Y_{i,r}Y_{i,s})

    Args:
        num_qubits (int): Total number of qubits in the system (N*K).
        N (int): The number of nodes in the original graph.
        K (int): The number of communities.

    Returns:
        qiskit.quantum_info.SparsePauliOp: The constructed XY Mixer Hamiltonian.
    """
    mixer_pauli_terms = []

    for i in range(N):
        for r in range(K):
            for s in range(r + 1, K):
                qubit_idx_ir = i * K + r  # Qubit for node i in community r
                qubit_idx_is = i * K + s  # Qubit for node i in community s

                for pauli in ('X', 'Y'):
                    pauli_string = ['I'] * num_qubits
                    pauli_string[qubit_idx_ir] = pauli
                    pauli_string[qubit_idx_is] = pauli
                    mixer_pauli_terms.append(("".join(reversed(pauli_string)), 1.0))

    mixer_hamiltonian_xy = SparsePauliOp.from_list(mixer_pauli_terms, num_qubits=num_qubits)

    print(f"\n\nXY Mixer Hamiltonian constructed for {num_qubits} qubits:")
    print(mixer_hamiltonian_xy)
    print(f"Number of terms in XY Mixer Hamiltonian: {len(mixer_pauli_terms)}")

    return mixer_hamiltonian_xy


def create_generator_based_x_mixer(num_qubits, N, codeword_length, generator_checks):
    """Creates the generator-based X mixer Hamiltonian for the Hamming encoding.

    For each row g_k of generator_checks, build a Pauli-X string supported on g_k
    (X at each support qubit, I everywhere else) and sum them:

        H_M^node = sum_k X^(x)g_k

    This block is repeated for every node, so the full mixer is
    H_M = sum_{i in V} H_M^node(i), giving len(generator_checks)*N terms in total.

    Why this mixer
    --------------
    X^(x)g_k acts on a computational basis state as |c> -> |c XOR g_k>.  Every
    generator row is itself a codeword, so XOR-ing it with a codeword yields
    another codeword: the mixer moves amplitude *within* the code space of each
    node and never leaks into the invalid (non-codeword) strings, leaving the
    parity penalty nothing to fight.

    Reachability caveat
    -------------------
    The rows must *span* the code to reach all 16 codewords per node.
    Hamming(8,4) is self-dual, so its 4 parity rows do span it.  The 3 parity
    rows of Hamming(7,4) span only 8 of the 16 codewords, so pass the 4
    generator rows (GENERATOR_CODES_74) for full coverage.

    All terms are pure X-strings, so they mutually commute and
    exp(-i b H_M) factorises exactly into the individual rotations -- no
    Trotter error is introduced by the mixer layer.

    Initial state
    -------------
    This mixer only preserves the code space if the circuit *starts* inside it;
    the default |+>^n would leak immediately.  Pair it with the Steane logical
    |+_L> from build_steane_code_initial_state, selected by name in
    create_qaoa_ansatz(..., initial_state="steane", parity_checks=...).

    Qubit mapping: node i, bit q  ->  i * codeword_length + q

    Args:
        num_qubits (int): Total number of qubits (N * codeword_length).
        N (int): Number of nodes in the graph.
        codeword_length (int): Qubits per node / codeword length (7 or 8).
        generator_checks (list(list(int))): Code rows to use as mixer atoms,
            each given as 0-indexed bit positions, e.g. GENERATOR_CODES_74.

    Returns:
        qiskit.quantum_info.SparsePauliOp: The constructed generator-based X mixer.
    """
    mixer_pauli_terms = []

    for i in range(N):
        for g in generator_checks:  # One X-string per code row g_k
            pauli_string_x = ['I'] * num_qubits
            for q in g:  # Place X at every support position of this generator
                if q >= codeword_length:
                    raise ValueError(
                        f"Code row support {q} exceeds codeword length {codeword_length}."
                    )
                pauli_string_x[i * codeword_length + q] = 'X'
            mixer_pauli_terms.append(("".join(reversed(pauli_string_x)), 1.0))

    mixer_hamiltonian_x = SparsePauliOp.from_list(mixer_pauli_terms, num_qubits=num_qubits)

    print(f"\n\nGenerator-based X Mixer Hamiltonian constructed for {num_qubits} qubits:")
    print(mixer_hamiltonian_x)
    print(f"Number of terms in X Mixer Hamiltonian: {len(mixer_pauli_terms)}")

    return mixer_hamiltonian_x
