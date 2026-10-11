"""IBM Quantum hardware execution: credentials, backend selection, transpilation, SPSA and sampling.

Credentials are never hard-coded: they are read from .env (gitignored) or the
process environment.  See .env.example for the expected keys.
"""

import os
from pathlib import Path

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from quantumUtility.simulator import to_bitstring

# Project-root .env, so scripts work from any working directory
DEFAULT_ENV_PATH = str(Path(__file__).resolve().parent.parent / ".env")

# Values from .env.example that mean "not filled in yet"
_PLACEHOLDERS = {"your_ibm_quantum_api_key_here", "your_instance_crn_here"}


def load_ibm_credentials(env_path=DEFAULT_ENV_PATH):
    """Read the IBM Quantum credentials from .env (or the process environment).

    Keeping the token out of the code means it never lands in git or in a
    committed output cell.  .env is gitignored; copy .env.example to create it.

    Recognised keys (environment variables win over the file):
        IBM_KEY       API token from https://quantum.cloud.ibm.com  (required)
        CRN           IBM Cloud instance CRN ("crn:v1:bluemix:...")
        IBM_INSTANCE  alternative name for the instance, if CRN is not set
        IBM_CHANNEL   overrides the auto-detected channel

    Channel selection: the legacy "ibm_quantum" channel (IBM Quantum Platform
    classic) was retired, so an instance is now always a CRN and the channel is
    "ibm_cloud" on qiskit-ibm-runtime < 0.30.  The channel is inferred from the
    instance string so a CRN never gets paired with the dead classic endpoint.

    Args:
        env_path (str): Path to the .env file. Defaults to the project root .env.

    Returns:
        tuple: (token (str), channel (str), instance (str)).

    Raises:
        RuntimeError: If no token or no instance is configured, or either is
            still the placeholder value.
    """
    file_vals = {}
    if os.path.exists(env_path):
        with open(env_path) as env_file:
            for line in env_file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    file_vals[key.strip()] = val.strip().strip('"').strip("'")

    def get(key):
        value = os.environ.get(key) or file_vals.get(key)
        return None if value in _PLACEHOLDERS else value

    token = get("IBM_KEY")
    if not token:
        raise RuntimeError(
            f"No IBM Quantum token found. Put your API token in {env_path} as\n"
            f"    IBM_KEY=<token from https://quantum.cloud.ibm.com>"
        )

    instance = get("CRN") or get("IBM_INSTANCE")
    if not instance:
        raise RuntimeError(
            f"No instance found. Put your instance CRN in {env_path} as\n"
            f"    CRN=crn:v1:bluemix:public:quantum-computing:...\n"
            f"(copy it from the instance list at https://quantum.cloud.ibm.com)"
        )

    # A CRN is an IBM Cloud instance; only the retired classic platform used
    # hub/group/project strings, so anything else is assumed to be that.
    channel = get("IBM_CHANNEL") or ("ibm_cloud" if instance.startswith("crn:") else "ibm_quantum")

    return token, channel, instance


def connect_ibm_service(env_path=DEFAULT_ENV_PATH):
    """Connect to IBM Quantum using the credentials in .env.

    Note on channel names: qiskit-ibm-runtime 0.24 accepts only "ibm_quantum"
    or "ibm_cloud".  "ibm_quantum_platform" is the name used by runtime >= 0.30
    and is rejected here, so it must not be hard-coded.

    The instance is deliberately not printed, so it cannot end up in a saved
    notebook output or log.

    Args:
        env_path (str): Path to the .env file holding IBM_KEY and CRN.

    Returns:
        QiskitRuntimeService: An authenticated service handle.
    """
    token, channel, instance = load_ibm_credentials(env_path)
    print(f"Connecting to IBM Quantum (channel={channel}, instance from {env_path})")
    return QiskitRuntimeService(channel=channel, instance=instance, token=token)


def select_ibm_backend(service, num_qubits):
    """Pick the least-busy real QPU with room for the circuit.

    - simulator=False keeps us off the cloud simulators; we want real hardware.
    - operational=True skips devices down for maintenance.

    Args:
        service (QiskitRuntimeService): Authenticated service handle.
        num_qubits (int): Minimum qubit count the circuit needs.

    Returns:
        IBMBackend: The selected backend.
    """
    ibm_backend = service.least_busy(
        min_num_qubits=num_qubits,
        simulator=False,
        operational=True,
    )
    status = ibm_backend.status()
    print(f"Selected backend: {ibm_backend.name} "
          f"({ibm_backend.num_qubits} qubits, {status.pending_jobs} jobs queued)")
    return ibm_backend


def transpile_for_hardware(circuit, ibm_backend, strip_measurements, optimization_level=3):
    """Transpile a circuit to a backend's native gate set and connectivity.

    optimization_level=3 is the most thorough: it searches for a good qubit
    layout (minimising SWAPs), cancels gates, and cuts depth.  Depth drives
    noise on real hardware, so the aggressive setting is worth the compile time.

    Measurements: which way you want them depends on the primitive.
        strip_measurements=True   for an Estimator.  Estimators reject circuits
            containing measure instructions, so they must be removed first.
        strip_measurements=False  for a Sampler.  Transpiling the *measured*
            circuit lets the transpiler route each measurement to whichever
            physical qubit the layout chose, while the classical bits keep
            their original order.  Clbit j then still means logical qubit j,
            which is what the decoder assumes.

    Never add measure_all() to an already-transpiled circuit: that measures
    physical qubit j into clbit j, and optimization_level=3 permutes the
    layout, so the decoded bitstring is scrambled.

    Args:
        circuit (QuantumCircuit): The logical circuit.
        ibm_backend (IBMBackend): Target backend.
        strip_measurements (bool): True for Estimator use, False for Sampler use.
        optimization_level (int): Transpiler effort, 0-3. Default 3.

    Returns:
        QuantumCircuit: The ISA (Instruction Set Architecture) circuit: native
            gates, mapped to physical qubits, with measurements if requested.
    """
    if strip_measurements:
        circuit = circuit.remove_final_measurements(inplace=False)
    elif circuit.num_clbits == 0:
        # Sampler needs measurements; add them *before* transpiling, not after.
        circuit = circuit.copy()
        circuit.measure_all()

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=ibm_backend,
        seed_transpiler=123,
    )
    isa_circuit = pass_manager.run(circuit)
    print(f"Transpiled depth: {isa_circuit.depth()} (logical: {circuit.depth()}), "
          f"2q gates: {sum(1 for inst in isa_circuit.data if inst.operation.num_qubits == 2)}")
    return isa_circuit


def estimate_circuit_fidelity(isa_circuit, ibm_backend):
    """Rough fidelity estimate for an ISA circuit, from the backend's calibration.

    Multiplies (1 - error) over every gate and readout in the circuit using the
    device's *median* error rates.  This ignores crosstalk, idling decoherence
    and the actual per-qubit spread, so it is an optimistic upper bound -- if
    this number is already tiny, the real run will be worse.

    It is worth checking before submitting: a deep QAOA ansatz on a few dozen
    qubits can easily land at ~1e-4, i.e. the measured distribution is noise and
    the SPSA optimiser would be descending on nothing.

    Args:
        isa_circuit (QuantumCircuit): The transpiled (ISA) circuit.
        ibm_backend (IBMBackend): The backend it was transpiled for.

    Returns:
        float: Estimated probability that the whole circuit runs error-free.
    """
    target = ibm_backend.target

    # Median error for each gate type in the circuit, ignoring missing entries
    def median_error(gate_name):
        if gate_name not in target:
            return None
        errs = [target[gate_name][q].error for q in target[gate_name]
                if target[gate_name][q] is not None and target[gate_name][q].error is not None]
        return float(np.median(errs)) if errs else None

    fidelity = 1.0
    counts = dict(isa_circuit.count_ops())
    print(f"\n{'Gate':<10} {'Count':<8} {'Median err':<12} Contribution")
    print("-" * 48)
    for gate, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        err = median_error(gate)
        if err is None:          # barrier, measure handled below, etc.
            continue
        contribution = (1.0 - err) ** count
        fidelity *= contribution
        print(f"{gate:<10} {count:<8} {err:<12.3e} {contribution:.3e}")

    # Readout, one per qubit that is actually used.  Not num_qubits: an ISA
    # circuit is padded out to the full device width and counting every idle
    # wire would inflate the readout term.  Not num_clbits either: the
    # Estimator path has its measurements stripped, so that is 0.
    ro = median_error("measure")
    if ro is not None:
        active = {isa_circuit.find_bit(q).index
                  for inst in isa_circuit.data for q in inst.qubits}
        n_ro = len(active)
        contribution = (1.0 - ro) ** n_ro
        fidelity *= contribution
        print(f"{'readout':<10} {n_ro:<8} {ro:<12.3e} {contribution:.3e}")

    print("-" * 48)
    print(f"{'TOTAL':<10} {'':<8} {'':<12} {fidelity:.3e}")
    return fidelity


def spsa_step(params, isa_ansatz, isa_hamiltonian, estimator, tracker,
              learning_rate=0.1, perturbation=0.1, rng=None):
    """One SPSA step, costing exactly ONE hardware job.

    SPSA (Simultaneous Perturbation Stochastic Approximation, 1992) approximates
    the gradient from two probes either side of the current point, whatever the
    number of parameters:

        g ~= (f(t + c*D) - f(t - c*D)) / (2c) * D^-1,   D a random +/-1 vector

    Both probes are submitted as a single Estimator pub with a (2, n) parameter
    array, so the two expectation values come back from one job instead of two.
    On the Open plan there are no Sessions, so every job queues separately --
    halving the job count halves the wall-clock time.

    D contains only +/-1, so D^-1 == D elementwise.

    Cost tracking is free here: the midpoint (f_plus + f_minus)/2 estimates the
    cost at the current point, so no third evaluation is needed.

    Args:
        params (np.ndarray): Current QAOA parameters.
        isa_ansatz (QuantumCircuit): Transpiled ansatz, measurements stripped.
        isa_hamiltonian (SparsePauliOp): Cost Hamiltonian already mapped through
            isa_ansatz.layout via apply_layout.
        estimator (EstimatorV2): Runtime Estimator bound to the backend.
        tracker (list): Collects the cost at each step, for plotting.
        learning_rate (float): SPSA step size for this iteration.
        perturbation (float): SPSA probe distance for this iteration.
        rng (np.random.Generator): Optional generator, for reproducible runs.

    Returns:
        tuple: (new_params (np.ndarray), cost (float)).
    """
    rng = np.random.default_rng() if rng is None else rng
    delta = rng.choice([-1.0, 1.0], size=len(params))

    # Both probes in ONE pub -> ONE job
    probes = np.array([params + perturbation * delta,
                       params - perturbation * delta])
    result = estimator.run([(isa_ansatz, isa_hamiltonian, probes)]).result()[0]
    f_plus, f_minus = (float(v) for v in result.data.evs)

    gradient = (f_plus - f_minus) / (2.0 * perturbation) * delta
    new_params = params - learning_rate * gradient

    cost = 0.5 * (f_plus + f_minus)   # free estimate of f at the current point
    tracker.append(cost)
    return new_params, cost


def run_spsa(isa_ansatz, isa_hamiltonian, ibm_backend, initial_params,
             n_iterations=20, shots=4096, seed=None):
    """Optimise QAOA parameters on hardware with SPSA, one job per iteration.

    Args:
        isa_ansatz (QuantumCircuit): Transpiled ansatz, measurements stripped.
        isa_hamiltonian (SparsePauliOp): Hamiltonian mapped through isa_ansatz.layout.
        ibm_backend (IBMBackend): The backend to run on.
        initial_params (list[float]): Starting values, in isa_ansatz.parameters order.
        n_iterations (int): Number of SPSA steps (= jobs). Default 20.
        shots (int): Shots per expectation value. Default 4096.
        seed (int): Optional seed for the SPSA perturbation directions.

    Returns:
        tuple: (params (np.ndarray), tracker (list[float])).
    """
    estimator = Estimator(mode=ibm_backend)
    estimator.options.default_shots = shots
    estimator.options.resilience_level = 1   # readout error mitigation (does NOT help gate errors)

    print(f"\nRunning SPSA: {n_iterations} iterations = {n_iterations} jobs "
          f"({shots} shots each)")

    rng = np.random.default_rng(seed)
    params = np.array(initial_params, dtype=float)
    tracker = []
    for i in range(n_iterations):
        # Standard SPSA decay schedule: both step size and probe distance shrink
        lr = 0.1 / (1 + i * 0.05)
        c = 0.1 / (1 + i * 0.01) ** 0.167
        params, cost = spsa_step(params, isa_ansatz, isa_hamiltonian, estimator,
                                 tracker, learning_rate=lr, perturbation=c, rng=rng)
        print(f"  iter {i+1:3d}/{n_iterations}  <H_C> = {cost:.4f}")

    return params, tracker


def run_qaoa_hardware(circuit, cost_hamiltonian, initial_params,
                      n_iterations=20, shots=4096, env_path=DEFAULT_ENV_PATH, seed=None):
    """Run QAOA on the least-busy IBM QPU, optimising with SPSA.

    Steps
    -----
    1. Connect via credentials in .env (never hard-code the token).
    2. Pick the least-busy real device with enough qubits.
    3. Transpile to that device, measurements stripped for the Estimator.
    4. Map the Hamiltonian through the chosen layout, ONCE.
    5. Run SPSA -- one job per iteration.

    Job budget (Open plan has no Sessions, so each job queues on its own):
        n_iterations jobs total, plus one more for the final sample_output_hardware.

    Args:
        circuit (QuantumCircuit): The logical QAOA ansatz (measurements optional).
        cost_hamiltonian (SparsePauliOp): The problem Hamiltonian.
        initial_params (list[float]): Starting values, in circuit.parameters order.
        n_iterations (int): Number of SPSA steps. Default 20.
        shots (int): Shots per expectation value. Default 4096.
        env_path (str): Path to the .env holding IBM_KEY and CRN.
        seed (int): Optional seed for the SPSA perturbation directions.

    Returns:
        tuple:
            - best_params (np.ndarray): Optimised parameters.
            - tracker (list[float]): Cost at each iteration.
            - isa_ansatz (QuantumCircuit): The transpiled circuit.
            - ibm_backend (IBMBackend): The backend used.
    """
    service = connect_ibm_service(env_path)
    ibm_backend = select_ibm_backend(service, circuit.num_qubits)

    isa_ansatz = transpile_for_hardware(circuit, ibm_backend, strip_measurements=True)
    # The layout is fixed after transpilation, so map the observable once.
    isa_hamiltonian = cost_hamiltonian.apply_layout(isa_ansatz.layout)

    params, tracker = run_spsa(isa_ansatz, isa_hamiltonian, ibm_backend, initial_params,
                               n_iterations=n_iterations, shots=shots, seed=seed)
    return params, tracker, isa_ansatz, ibm_backend


def sample_output_hardware(circuit, ibm_backend, num_qubits, optimal_params=None,
                           shots=4096):
    """Sample the optimised circuit on hardware and return the most likely bitstring.

    Why this exists separately from simulator.sample_output
    -------------------------------------------------------
    sample_output() re-adds measure_all() to an already-transpiled circuit.
    That is safe on AerSimulator, which has a trivial layout, but wrong on a
    QPU: optimization_level=3 permutes the layout, so clbit j ends up holding
    physical qubit j rather than logical qubit j.

    Here the LOGICAL circuit is transpiled with its measurements attached, so
    the transpiler routes each measurement to the right physical qubit while
    the classical bits keep logical order.

    Args:
        circuit (QuantumCircuit): The LOGICAL ansatz (not yet transpiled).
        ibm_backend (IBMBackend): Backend to run on.
        num_qubits (int): Logical qubit count.
        optimal_params (array-like): Parameter values to bind, if the circuit is
            still parameterised.
        shots (int): Number of shots. Default 4096.

    Returns:
        tuple:
            - bitstring (list[int]): Most likely bitstring, index j = logical qubit j.
            - probability (float): Its observed probability over the shots.
    """
    if circuit.num_parameters:
        if optimal_params is None:
            raise ValueError(
                f"circuit still has {circuit.num_parameters} unbound parameter(s); "
                f"pass optimal_params (the array returned by run_spsa)."
            )
        circuit = circuit.assign_parameters(optimal_params)

    # Sampler path: keep the measurements and transpile them along
    isa_circuit = transpile_for_hardware(circuit, ibm_backend, strip_measurements=False)

    sampler = Sampler(mode=ibm_backend)
    result = sampler.run([isa_circuit], shots=shots).result()[0]

    counts_int = result.data.meas.get_int_counts()
    total = sum(counts_int.values())
    distribution = {k: v / total for k, v in counts_int.items()}

    most_likely = max(distribution, key=distribution.get)
    bitstring = to_bitstring(most_likely, num_qubits)
    bitstring.reverse()   # np.binary_repr is MSB-first; index j -> qubit j

    print(f"Most likely bitstring: {bitstring}  "
          f"(probability {distribution[most_likely]:.4f} of {total} shots)")
    return bitstring, distribution[most_likely]
