# Thesis Project — Quantum Community Detection

MSc thesis implementation of graph community detection using the Quantum Approximate Optimisation Algorithm (QAOA). Three community assignment encodings are explored and compared: **one-hot**, **binary**, and **Hamming error-correcting codes**. Each encoding is implemented both as a full QAOA quantum pipeline (simulated via Qiskit Aer) and as a classical baseline optimiser (COBYLA or simulated annealing) for benchmarking.

---

## Repository Structure

```
Thesis-Project/
├── communityGraphFunctions.ipynb   # Graph construction and shared graph utilities
├── communityOneHotEncoding.ipynb   # One-hot encoding: QAOA + classical COBYLA
├── communityBinaryEncoding.ipynb   # Binary encoding: QAOA + classical COBYLA
├── communityHamming.ipynb          # Hamming(7,4) and Hamming(8,4): QAOA + simulated annealing
├── quantumUtility.ipynb            # Shared QAOA circuit and optimiser utilities
├── classicalUtility.ipynb          # Classical utility functions
├── requirements.txt                # Python environment dependencies
└── README.md                       # This file
```

The `.ipynb` notebooks are converted to `.py` scripts at runtime (via `nbconvert`) so that functions defined in one notebook can be imported by others.

---

## Notebooks

### `communityGraphFunctions.ipynb`
Shared graph construction and utility functions used across all encoding notebooks.

| Function | Description |
|---|---|
| `create_initial_graph(n)` | Creates a basic graph with two 3-node cliques connected by a bridge |
| `create_clique_chain_graph(x, y)` | Creates y cliques of size x connected in a cycle |
| `get_les_miserables_graph()` | Loads the Les Misérables co-appearance benchmark graph |
| `get_karate_club_graph()` | Loads Zachary's Karate Club benchmark graph |
| `get_davis_southern_women_graph()` | Loads the Davis Southern Women bipartite benchmark graph |
| `create_modularity_matrix(G, N)` | Computes the Newman modularity matrix B |
| `create_supernode(G, K)` | Constructs the supernode graph for the chosen encoding |
| `display_graph(G, ...)` | Visualises a NetworkX graph with spring layout |
| `spectral_stop(B)` | Checks the spectral stopping criterion (max eigenvalue of B) |

---

### `quantumUtility.ipynb`
Shared QAOA circuit construction, transpilation, optimisation, and sampling utilities.

| Function | Description |
|---|---|
| `create_total_cost_hamiltonian(...)` | Combines cost and penalty Hamiltonians into the total objective |
| `create_xy_mixer_hamiltonian(...)` | Constructs the XY mixer Hamiltonian for one-hot QAOA |
| `build_w_state_initial_state(...)` | Prepares the W-state initial state for one-hot encoding |
| `create_qaoa_ansatz(...)` | Builds the parameterised QAOA circuit ansatz |
| `simulator_backend(method)` | Returns an AerSimulator configured for statevector or MPS simulation |
| `run_ansatz_simulator_circuit_prep(...)` | Transpiles the circuit for local AerSimulator execution |
| `run_ansatz_qc_circuit_prep(...)` | Transpiles the circuit for a real IBM Quantum backend |
| `cost_fun_estimator(...)` | Evaluates the cost function using EstimatorV2 |
| `run_optimizer_simulator(...)` | Runs COBYLA optimisation on the simulator |
| `run_qaoa_hardware(...)` | Runs QAOA on a real IBM Quantum backend using SPSA |
| `sample_output(...)` | Samples bitstring counts from the optimised circuit |
| `plot_optimizer_results(...)` | Plots the cost function value over optimisation iterations |
| `group_nodes_by_community(...)` | Decodes a bitstring and colour-codes graph nodes by community |

---

### `communityOneHotEncoding.ipynb`
Community detection using **one-hot encoding**: each node is assigned exactly one of K qubits set to \|1⟩.

**QAOA pipeline:**
| Function | Description |
|---|---|
| `create_cost_modularity_hamiltonian(...)` | Constructs the modularity cost Hamiltonian H_mod |
| `create_onehot_penalty_hamiltonian(...)` | Constructs the one-hot constraint penalty Hamiltonian |
| `create_total_cost_hamiltonian(...)` | Combines cost and penalty into the total Hamiltonian |
| `run_one_hot(G, N, K, P, LAMBDA_PENALTY)` | Full QAOA pipeline entry point |

**Classical baseline (COBYLA):**
| Function | Description |
|---|---|
| `classical_objective(...)` | Modularity + one-hot penalty objective (continuous relaxation) |
| `run_classical_cobyla(...)` | COBYLA optimisation with multiple random restarts |
| `decode_assignments(...)` | Decodes continuous solution to integer community assignments |
| `plot_classical_convergence(...)` | Plots COBYLA objective over function evaluations |
| `plot_classical_communities(...)` | Visualises community assignments on the graph |
| `onehot_classical_optimization()` | Full classical pipeline entry point |

---

### `communityBinaryEncoding.ipynb`
Community detection using **binary encoding**: each node's community label is stored as a `ceil(log2(K))`-bit binary integer.

**QAOA pipeline:**
| Function | Description |
|---|---|
| `create_cost_hamiltonian(...)` | Constructs the binary-encoded modularity cost Hamiltonian |
| `group_nodes_by_community(...)` | Decodes binary bitstring and visualises communities |
| `binary_quantum_optimization(...)` | Full QAOA pipeline entry point |

**Classical baseline (COBYLA):**
| Function | Description |
|---|---|
| `build_modularity_matrix_binary(...)` | Constructs the modularity matrix B |
| `binary_objective(...)` | Binary modularity objective (continuous relaxation) |
| `run_binary_cobyla(...)` | COBYLA optimisation with multiple random restarts and box constraints |
| `decode_binary_assignments(...)` | Rounds continuous solution and decodes to community assignments |
| `plot_binary_convergence(...)` | Plots COBYLA objective over function evaluations |
| `plot_binary_communities(...)` | Visualises community assignments on the graph |
| `binary_classical_optimization(...)` | Full classical pipeline entry point |

---

### `communityHamming.ipynb`
Community detection using **Hamming error-correcting codes**. Two variants are implemented:

- **Hamming(7,4)** — 7-bit codewords, 4 data bits, corrects single-bit errors
- **Hamming(8,4)** — 8-bit codewords, 4 data bits + overall parity, corrects single-bit errors and detects double-bit errors

**QAOA pipeline (shared):**
| Function | Description |
|---|---|
| `create_modularity_penalty_hamiltonian(...)` | Constructs the modularity Hamiltonian for Hamming encoding |
| `create_parity_penalty_hamiltonian(...)` | Constructs the parity constraint penalty Hamiltonian |
| `sample_output(...)` | Samples circuit, applies syndrome correction, decodes communities |
| `run_Hamming_74(...)` | Full QAOA pipeline entry point for Hamming(7,4) |
| `run_Hamming_84(...)` | Full QAOA pipeline entry point for Hamming(8,4) |

**Classical baseline — Hamming(7,4) (simulated annealing):**
| Function | Description |
|---|---|
| `build_modularity_matrix_hamming(G)` | Constructs the modularity matrix B |
| `hamming_objective(...)` | Total objective H_total = −H_mod + parity penalty |
| `run_binary_sa(...)` | Simulated annealing over binary strings with Metropolis acceptance |
| `syndrome_correct_74(...)` | Hamming(7,4) single-bit error correction via syndrome decoding |
| `decode_hamming_assignments(...)` | Decodes corrected codewords to community assignments |
| `run_hamming74_classical(...)` | Full classical pipeline entry point for Hamming(7,4) |

**Classical baseline — Hamming(8,4) (simulated annealing):**
| Function | Description |
|---|---|
| `syndrome_correct_84(...)` | Hamming(8,4) syndrome correction with double-bit error detection |
| `decode_hamming_assignments_84(...)` | Decodes corrected codewords to community assignments |
| `run_hamming84_classical(...)` | Full classical pipeline entry point for Hamming(8,4) |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

> **Note:** `qiskit-aer==0.14.2` requires Python 3.8–3.11. On Python 3.12+ drop the version pins and let pip resolve compatible versions.

---

## Key Parameters

| Parameter | Description |
|---|---|
| `N` | Number of nodes in the graph |
| `K` | Number of communities |
| `P` | Number of QAOA layers (circuit depth) |
| `LAMBDA_PENALTY` | Penalty strength enforcing encoding constraints |
| `INITIAL_GAMMA` | Initial QAOA phase separation angle (default: π) |
| `INITIAL_BETA` | Initial QAOA mixing angle (default: π/2) |
