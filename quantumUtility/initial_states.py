"""Constraint-preserving QAOA initial states: W states (one-hot) and Steane |+_L> (Hamming)."""

import numpy as np
from qiskit.circuit import QuantumCircuit


def build_w_state_initial_state(num_qubits, N, K):
    """Prepare the W-state on each node's K-qubit block.

    For node i, the K qubits are [i*K, i*K+1, ..., i*K+K-1].
    Each block is initialised to:
        |W_K> = (1/sqrt(K))(|10...0> + |01...0> + ... + |00...1>)

    Why this is the right initial state
    ------------------------------------
    The XY mixer e^{-ib H_M} preserves Hamming weight within each
    node's qubit block.  The W-state has Hamming weight exactly 1,
    which is the one-hot constraint.  Starting here guarantees the
    circuit never leaves the valid (one-hot) subspace, so the
    penalty Hamiltonian is not fighting the initial state.

    Circuit construction (per node block, K qubits)
    ------------------------------------------------
    1.  X on qubit 0  ->  |1, 0, ..., 0>
    2.  For i = 0 to K-2:
          a. CRy(2*arccos(1/sqrt(K-i)))  control=qubits[i], target=qubits[i+1]
             Splits amplitude: 1/(K-i) stays at i, (K-i-1)/(K-i) moves to i+1
          b. CX  control=qubits[i+1], target=qubits[i]
             Disentangles the control qubit, moving it back to |0>

    Verification for K=3:
        Start:                              |100>
        After CRy(2arccos(1/sqrt(3))) + CX: 1/sqrt(3)|100> + sqrt(2/3)|010>
        After CRy(pi/2) + CX:              1/sqrt(3)|100> + 1/sqrt(3)|010> + 1/sqrt(3)|001>

    Args:
        num_qubits (int): Total qubits in the system (N*K).
        N (int): Number of nodes.
        K (int): Number of communities.

    Returns:
        QuantumCircuit: The initial-state preparation circuit (no measurements).
    """
    qc = QuantumCircuit(num_qubits)

    for node in range(N):
        # Qubit indices for this node's K community slots
        qubits = [node * K + r for r in range(K)]

        # Place the excitation on the first qubit of this node's block
        qc.x(qubits[0])

        # Spread it uniformly across all K positions
        for i in range(K - 1):
            # After i steps, (K-i) positions remain to share the amplitude.
            # Rotation angle places 1/(K-i) of amplitude at qubits[i].
            angle = 2.0 * np.arccos(1.0 / np.sqrt(K - i))
            qc.cry(angle, qubits[i], qubits[i + 1])  # controlled-Ry
            qc.cx(qubits[i + 1], qubits[i])           # un-entangle control qubit

    return qc


def build_steane_code_initial_state(num_qubits, N, codeword_length, parity_checks):
    """Prepare the logical |+_L> state on each node's codeword block.

    For node i, the L qubits are [i*L, i*L+1, ..., i*L+L-1] where
    L = codeword_length.  Each block is initialised to:

        |+_L> = (|0_L> + |1_L>)/sqrt(2) = (1/4) sum_{c in C} |c>

    i.e. the uniform superposition over all 16 codewords c of the Hamming
    code C, every one carrying amplitude 1/4 = 1/sqrt(16).

    Why this is the right initial state
    ------------------------------------
    The generator-based X mixer e^{-ib H_M} maps codewords to codewords, so
    it can only stay inside the code space if the circuit starts there.
    |+_L> is the maximal-entropy state *within* that space: it gives equal
    weight to all 16 valid community labels for the node and zero weight to
    the 112 invalid strings.  This is the code-space analogue of the usual
    |+>^n QAOA start, and it leaves the parity penalty nothing to fight.
    (|0_L> alone is the uniform superposition over the 8 even-weight
    codewords; adding |1_L>, its complement coset, completes C.)

    Circuit construction (per node block, L qubits)
    ------------------------------------------------
    Parity bits sit where (q+1) is a power of 2 -- check j owns bit 2^j - 1
    -- and the remaining 4 positions are the data bits:

        Hamming(7,4): parity @ {0,1,3},   data @ [2,4,5,6]
        Hamming(8,4): parity @ {0,1,3,7}, data @ [2,4,5,6]

    1.  H on each of the 4 data qubits
            -> (1/4) sum_x |x on data bits, 0 on parity bits>
    2.  For each check P_j, CX from every other qubit of P_j onto its parity
        bit 2^j - 1, writing that bit as the XOR of the rest.  This is exactly
        the even-parity condition, so each branch |x> lands on its codeword.

    Checks are applied in order, so bits 0,1,3 are already final when the
    Hamming(8,4) overall-parity check P4 reads them.

    Verification for Hamming(7,4):
        P1={0,2,4,6} -> b0 = x2 XOR x4 XOR x6
        P2={1,2,5,6} -> b1 = x2 XOR x5 XOR x6
        P3={3,4,5,6} -> b3 = x4 XOR x5 XOR x6
        The 16 branches are distinct, so the result is (1/4) sum_{c in C} |c>.

    Qubit mapping: node i, bit q  ->  i * codeword_length + q

    Args:
        num_qubits (int): Total qubits in the system (N * codeword_length).
        N (int): Number of nodes.
        codeword_length (int): Qubits per node / codeword length (7 or 8).
        parity_checks (list(list(int))): Parity check sets for Hamming(7,4)
            or Hamming(8,4), e.g. PARITY_CODES_74.

    Returns:
        QuantumCircuit: The initial-state preparation circuit (no measurements).
    """
    qc = QuantumCircuit(num_qubits)

    # Check j owns parity bit 2^j - 1; every other position is a data bit.
    parity_bits = [2 ** j - 1 for j in range(len(parity_checks))]
    for j, (P, p) in enumerate(zip(parity_checks, parity_bits)):
        if p not in P:
            raise ValueError(
                f"Parity check {j} ({P}) does not cover its parity bit {p}; "
                f"expected the standard Hamming layout."
            )
    data_bits = [q for q in range(codeword_length) if q not in parity_bits]

    for node in range(N):
        # Qubit indices for this node's L codeword slots
        qubits = [node * codeword_length + q for q in range(codeword_length)]

        # Uniform superposition over the 4 data bits: all 16 messages at once
        for q in data_bits:
            qc.h(qubits[q])

        # Write each parity bit as the XOR of the other qubits in its check,
        # steering every branch onto a valid codeword
        for P, p in zip(parity_checks, parity_bits):
            for q in P:
                if q != p:
                    qc.cx(qubits[q], qubits[p])

    return qc
