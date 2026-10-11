"""Local AerSimulator execution: transpilation, COBYLA optimisation and sampling."""

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize


def simulator_backend(method):
    """Returns a seeded AerSimulator for "statevector", "mps" or anything else (automatic)."""
    if method == "statevector":
        return AerSimulator(seed_simulator=123)
    elif method == "mps":
        return AerSimulator(seed_simulator=123,
                            method="matrix_product_state",
                            matrix_product_state_max_bond_dimension=256,
                            matrix_product_state_truncation_threshold=1e-16)
    else:
        return AerSimulator(seed_simulator=123, method="automatic")


def run_ansatz_simulator_circuit_prep(circuit, method="automatic"):
    """Prepares the quantum circuit for execution on a local AerSimulator.

    Uses generate_preset_pass_manager (post-Qiskit 0.44 Runtime-native API)
    instead of the legacy transpile() function. This ensures the returned
    circuit carries a proper .layout attribute, which EstimatorV2 requires
    via hamiltonian.apply_layout(ansatz.layout).

    Args:
        circuit (qiskit.circuit.QuantumCircuit): The quantum circuit to be prepared.
        method (str): Simulation method, see simulator_backend.

    Returns:
        tuple:
            - transpiled_qc (qiskit.circuit.QuantumCircuit): The transpiled circuit (no measurements).
            - backend (qiskit_aer.AerSimulator): The AerSimulator instance.
    """
    # Strip measurements before transpilation — HighLevelSynthesis cannot synthesise
    # measure instructions. EstimatorV2 handles expectation values without them.
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)

    backend = simulator_backend(method)

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,   # Level 1 is sufficient for simulation — fast
        backend=backend,
        seed_transpiler=123,
    )
    transpiled_qc = pass_manager.run(circuit_no_meas)
    return transpiled_qc, backend


def cost_fun_estimator(params, ansatz, hamiltonian, estimator, eval_tracker):
    """Calculates the expectation value of the Hamiltonian for a given ansatz and parameters.

    Serves as the objective function for the optimizer.

    Args:
        params (np.ndarray): The current QAOA parameters, in ansatz.parameters order.
        ansatz (qiskit.circuit.QuantumCircuit): The QAOA ansatz circuit.
        hamiltonian (qiskit.quantum_info.SparsePauliOp): The total Hamiltonian (objective function).
        estimator (qiskit_ibm_runtime.EstimatorV2): The Qiskit Runtime Estimator primitive.
        eval_tracker (list): A list to track the cost values during optimization.

    Returns:
        float: The expectation value (cost) of the Hamiltonian for the given parameters.
    """
    # Map the Hamiltonian onto the transpiled circuit's layout so the observable
    # and the ansatz act on the same (physical) qubits.
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    # A Pub (Primitive Unified Bloc) bundles ansatz, observable and parameters for one job.
    pub = (ansatz, isa_hamiltonian, params)
    results = estimator.run([pub]).result()[0]
    cost = float(results.data.evs)

    eval_tracker.append(cost)
    return cost


def run_optimizer_simulator(transpiled_qc, backend, initial_params, total_hamiltonian):
    """Runs COBYLA to find the QAOA parameters minimising the total Hamiltonian.

    Args:
        transpiled_qc (qiskit.circuit.QuantumCircuit): The transpiled QAOA ansatz circuit.
        backend (qiskit_aer.AerSimulator): The simulator backend.
        initial_params (list): Initial values, in transpiled_qc.parameters order
            (see quantumUtility.ansatz.initial_qaoa_params).
        total_hamiltonian (SparsePauliOp): The objective.

    Returns:
        tuple:
            - result (scipy.optimize.OptimizeResult): The result object from the optimization.
            - value_tracker (list): Cost values recorded during the optimization process.
    """
    value_tracker = []

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 4096
    estimator.options.seed_estimator = 123

    result = minimize(
        cost_fun_estimator,
        initial_params,
        args=(transpiled_qc, total_hamiltonian, estimator, value_tracker),
        method="COBYLA",
        tol=1e-2,
    )

    print(f"\n\nOptimizer Result: ")
    print(result)
    print(f"\n\nValue Tracker: ")
    print(value_tracker)

    return result, value_tracker


def to_bitstring(integer, num_bits):
    """Converts an integer to a list of bits (MSB first) of the given length."""
    return [int(digit) for digit in np.binary_repr(integer, width=num_bits)]


def sample_output(optimized_circuit_input, backend, num_qubits):
    """Samples an optimized circuit on the simulator and returns the most likely bitstring.

    Simulator only
    --------------
    This re-adds measure_all() to an already-transpiled circuit, which measures
    physical qubit j into clbit j.  AerSimulator has no coupling map, so its
    layout is trivial and physical == logical.  On a QPU the layout is permuted
    and the decoded bitstring would be scrambled -- use
    quantumUtility.hardware.sample_output_hardware() there instead.

    Args:
        optimized_circuit_input (qiskit.circuit.QuantumCircuit): Circuit with parameters bound.
        backend (qiskit_aer.AerSimulator): The simulator backend.
        num_qubits (int): Total number of qubits.

    Returns:
        list[int]: The most likely bitstring, index j = qubit j.
    """
    # The circuit may have had measurements stripped for EstimatorV2; re-add them.
    circuit = optimized_circuit_input.copy()
    if circuit.num_clbits == 0:
        circuit.measure_all()  # adds a classical register named 'meas'

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 10000

    job = sampler.run([circuit], shots=int(1e4))

    counts_int = job.result()[0].data.meas.get_int_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    print(final_distribution_int)

    most_likely = max(final_distribution_int, key=final_distribution_int.get)
    most_likely_bitstring = to_bitstring(most_likely, num_qubits)
    # Reverse to match Qiskit's qubit ordering (q0 is rightmost)
    most_likely_bitstring.reverse()

    print("Result bitstring:", most_likely_bitstring)

    return most_likely_bitstring
