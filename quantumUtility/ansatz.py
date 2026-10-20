"""QAOA ansatz construction and its initial parameters."""

import numpy as np
from qiskit.circuit.library import QAOAAnsatz

from quantumUtility.initial_states import build_steane_code_initial_state, build_w_state_initial_state

# Default starting angles for every QAOA layer
INITIAL_GAMMA = np.pi      # phase-separation (cost) angle
INITIAL_BETA = np.pi / 2   # mixing angle


def create_total_cost_hamiltonian(cost_hamiltonian, penalty_hamiltonian, num_qubits):
    """Combines the Cost Hamiltonian and the Penalty Hamiltonian to form the total objective function.

    The total Hamiltonian represents the overall optimization problem, where the goal is to
    find a quantum state that minimizes its expectation value. It includes terms that
    encourage desired community structures (Cost Hamiltonian) and terms that enforce
    the encoding constraints (Penalty Hamiltonian).

    Args:
        cost_hamiltonian (qiskit.quantum_info.SparsePauliOp): The modularity objective.
        penalty_hamiltonian (qiskit.quantum_info.SparsePauliOp): The encoding constraint penalty.
        num_qubits (int): Total number of qubits (for reporting).

    Returns:
        qiskit.quantum_info.SparsePauliOp: The combined total Hamiltonian.
    """
    total_hamiltonian = cost_hamiltonian + penalty_hamiltonian

    print(f"Total Hamiltonian constructed for {num_qubits} qubits:")
    print(total_hamiltonian)
    print(f"Number of terms in Total Hamiltonian: {len(total_hamiltonian)}")

    return total_hamiltonian


def create_qaoa_ansatz(total_hamiltonian, P, N, K, mixer_hamiltonian=None, initial_state=None,
                       parity_checks=None):
    """Creates a QAOA ansatz circuit with a constraint-preserving initial state.

    The initial state is chosen by name, and must match the mixer so that the
    two together keep the search inside the valid subspace:

        "W"       one W_K state per node's K-qubit block.  Hamming weight 1,
                  i.e. the one-hot constraint.  Pair with the XY mixer from
                  create_xy_mixer_hamiltonian, which preserves Hamming weight.
                  Only for One-Hot.

        "steane"  the logical |+_L> = (1/4) sum_{c in C} |c> per node's
                  codeword block, the uniform superposition over the 16 Hamming
                  codewords.  Pair with the generator-based X mixer from
                  create_generator_based_x_mixer, which maps codewords to
                  codewords.  Only for Hamming(7,4)/(8,4); needs parity_checks.

    Anything else is forwarded to QAOAAnsatz untouched, so a QuantumCircuit can
    still be passed directly, and None falls back to the default |+>^n.

    Args:
        total_hamiltonian (SparsePauliOp): The problem Hamiltonian (cost operator).
        P (int): The number of QAOA layers (reps).
        N (int): Number of nodes in the graph.
        K (int): Number of communities.
        mixer_hamiltonian (SparsePauliOp): The mixer Hamiltonian (XY or generator-based X).
        initial_state (str or QuantumCircuit): "W", "steane", an explicit circuit, or None.
        parity_checks (list(list(int))): Parity check sets for Hamming(7,4)
            or Hamming(8,4). Required for "steane".

    Returns:
        QuantumCircuit: The constructed QAOA ansatz circuit with measurements.
    """
    num_qubits = total_hamiltonian.num_qubits

    if initial_state == "W":  # Only for One-Hot
        initial_state = build_w_state_initial_state(num_qubits, N, K)

    elif initial_state == "steane":  # Only for Hamming(7,4)/(8,4)
        if parity_checks is None:
            raise ValueError('initial_state="steane" requires parity_checks '
                             '(e.g. PARITY_CODES_74 or PARITY_CODES_84).')
        # Each node occupies one codeword block, so L = num_qubits / N
        codeword_length = num_qubits // N
        initial_state = build_steane_code_initial_state(num_qubits, N, codeword_length,
                                                        parity_checks)

    circuit = QAOAAnsatz(cost_operator=total_hamiltonian, reps=P,
                         mixer_operator=mixer_hamiltonian,
                         initial_state=initial_state)
    circuit.measure_all()

    # Decompose into elementary gates that AerSimulator understands.
    circuit = circuit.decompose()
    return circuit


def initial_qaoa_params(circuit, initial_gamma=INITIAL_GAMMA, initial_beta=INITIAL_BETA):
    """Builds the initial parameter vector in the order the circuit expects.

    QAOAAnsatz names its parameters β[0..P-1] and γ[0..P-1], and Qiskit binds
    a plain list in the order of circuit.parameters, which is sorted by name:
    [β[0], ..., β[P-1], γ[0], ..., γ[P-1]].  A hand-built list such as
    [gamma, beta] * P therefore puts gamma into β[0].  Matching on the name
    puts each value on the right angle for any number of layers P.

    Args:
        circuit (QuantumCircuit): The (possibly transpiled) QAOA ansatz.
        initial_gamma (float): Starting value for every γ (cost) parameter.
        initial_beta (float): Starting value for every β (mixer) parameter.

    Returns:
        list[float]: One value per circuit parameter, in circuit.parameters order.
    """
    values = []
    for parameter in circuit.parameters:
        if parameter.name.startswith("γ"):
            values.append(initial_gamma)
        elif parameter.name.startswith("β"):
            values.append(initial_beta)
        else:
            raise ValueError(f"Unexpected parameter {parameter.name!r} in the QAOA ansatz.")
    return values
