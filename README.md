# Thesis Project — Quantum Community Detection

MSc thesis implementation of graph community detection using the Quantum Approximate Optimisation Algorithm (QAOA). Three community assignment encodings are explored and compared: **one-hot**, **binary**, and **Hamming error-correcting codes** (Hamming(7,4) and Hamming(8,4)). Each encoding is implemented both as a QAOA pipeline (simulated with Qiskit Aer, and for Hamming also on IBM Quantum hardware) and as a classical baseline (COBYLA or simulated annealing) for benchmarking.

---

## Repository Structure

```
Thesis-Project/
├── graphUtility/
│   ├── graphs.py            # Test graphs, NetworkX benchmarks, graph_from_spec, supernode graph
│   ├── modularity.py        # Weighted modularity matrix B, total edge weight m, spectral stop
│   └── plotting.py          # Graph / community / convergence / circuit plots (show or save)
├── quantumUtility/
│   ├── initial_states.py    # W state (one-hot), Steane |+_L> (Hamming)
│   ├── mixers.py            # XY mixer (one-hot), generator-based X mixer (Hamming)
│   ├── ansatz.py            # Total Hamiltonian, QAOA ansatz, initial parameters by name
│   ├── simulator.py         # AerSimulator transpilation, COBYLA optimisation, sampling
│   └── hardware.py          # IBM credentials from .env, backend selection, SPSA, fidelity estimate
├── quantumEncodings/
│   ├── onehot.py            # One-hot Hamiltonians, decoding, run_onehot_qaoa
│   ├── binary.py            # Binary Hamiltonian, decoding, run_binary_qaoa
│   └── hamming.py           # Hamming codes, Hamiltonians, syndrome decoding,
│                            #   run_hamming_qaoa, run_hamming_hardware
├── classicalEncoding/
│   ├── onehot.py            # COBYLA continuous relaxation
│   ├── binary.py            # COBYLA continuous relaxation
│   └── hamming.py           # Simulated annealing on the same energy as the QAOA Hamiltonian
├── scripts/
│   ├── run.py               # Run one experiment; saves plots + result.json
│   └── analysis.py          # Analyse a saved hardware run: correct / noise / repeat
├── test/
│   └── experiment_matrix.py # Runs every encoding x solver against every graph; status report
├── notebooks/archive/       # The original notebooks, unchanged
├── .env.example             # Template for IBM credentials (copy to .env)
└── pyproject.toml           # Project metadata and pinned dependencies
```

---

## Setup

Dependencies are declared in `pyproject.toml` (Python 3.10–3.12). Pick one:

```bash
# uv: no setup step, `uv run` creates .venv and installs everything on first use
uv run scripts/run.py --encoding onehot --solver classical --graph karate -K 2 --lam 30

# pip: one install for the dependencies and the project packages
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # add ".[notebooks]" for Jupyter (archived notebooks)
```

An existing environment that already has the dependencies (e.g. a conda env) can run `python scripts/run.py ...` directly without installing the project: the scripts add the project root to the import path.

For IBM Quantum hardware runs, copy `.env.example` to `.env` and fill in `IBM_KEY` and `CRN`. `.env` is gitignored, and credentials are never printed.

---

## Running experiments

All commands run from the project root. Plots and a `result.json` are saved to `results/<timestamp>_<encoding>_<solver>/` (gitignored); add `--show` to also display figures.

```bash
# QAOA on the simulator
python scripts/run.py --encoding onehot    --solver qaoa --graph clique-chain:3x8 -K 2 -P 1 --lam 1
python scripts/run.py --encoding binary    --solver qaoa --graph clique-chain:3x3 -K 3 -P 2
python scripts/run.py --encoding hamming74 --solver qaoa --graph clique-chain:3x2 -K 2 -P 1 --lam 0.05

# Classical baselines
python scripts/run.py --encoding onehot    --solver classical --graph davis            -K 2 --lam 30
python scripts/run.py --encoding hamming84 --solver classical --graph clique-chain:3x2 -K 2 --lam 0.05

# IBM Quantum hardware (Hamming only): dry run by default, --submit spends n_iterations + 1 jobs
python scripts/run.py --encoding hamming74 --solver hardware --graph clique-chain:3x2 -K 2 --lam 0.05
python scripts/run.py --encoding hamming74 --solver hardware --graph clique-chain:3x2 -K 2 --lam 0.05 --submit

# Analyse a submitted hardware run
python scripts/analysis.py correct results/<run>/result.json              # syndrome correction report, no jobs
python scripts/analysis.py noise   results/<run>/result.json              # uniform-noise hypothesis test, no jobs
python scripts/analysis.py repeat  results/<run>/result.json --n-repeats 10   # costs 10 jobs
```

Graph specs: `clique-chain:XxY` (Y cliques of size X in a cycle), `initial`, `karate`, `les-miserables`, `davis`.

The modules can also be imported directly, e.g. from a notebook in the project root:

```python
from graphUtility.graphs import create_clique_chain_graph
from quantumEncodings.hamming import run_hamming_qaoa, HAMMING_74

G, N = create_clique_chain_graph(3, 2)
result = run_hamming_qaoa(G, N, K=2, P=1, lambda_penalty=0.05, code=HAMMING_74)
```

---

## Experiment matrix

`test/experiment_matrix.py` runs each experiment row (binary, one-hot with K = 2/4/8/16, Hamming(7,4), Hamming(8,4); classical and QAOA) against each graph (Tri-2 … Tri-16 = 2–16 triangles in a cycle, Les Misérables, Karate Club, Davis Southern Women). Every cell runs `scripts/run.py` in its own process with a timeout and is scored:

- ✅ ran, every node has a valid community, and modularity ≥ 90% of a greedy-modularity reference with ≤ K communities
- ⚠️ ran, but the partition is poor, has invalid nodes, or no structure was found
- ❌ error, timeout, or more qubits than the simulator supports (statevector 29, MPS 63; not attempted)

Rows without a K in their name use the graph's planted K (number of triangles; 2 for the benchmarks).

```bash
python test/experiment_matrix.py                                  # all 112 cells
python test/experiment_matrix.py --rows "Cl H74" "Q H74" --cols Tri-2 Tri-3 --timeout 120
```

The status grid and per-cell details go to `test/results/<timestamp>/report.md` (plus `report.json` and each cell's plots, `result.json` and `run.log`).

---

## Key Parameters

| Parameter | Description |
|---|---|
| `N` | Number of nodes in the graph |
| `K` | Number of communities |
| `P` | Number of QAOA layers (circuit depth) |
| `--lam` / `lambda_penalty` | Penalty strength enforcing encoding constraints |
| `--gamma` / `--beta` | Initial QAOA angles for every layer (default π and π/2) |
| `--mixer` | Hamming mixer: `generator` (default for (7,4)) or `x` (default for (8,4)) |

Notes:
- Initial QAOA parameters are assigned by parameter name (`γ[i]`, `β[i]`), because Qiskit orders `circuit.parameters` as β before γ.
- The modularity matrix uses weighted degrees and `list(G.nodes())` order everywhere, for both QAOA and the classical baselines.
- The Hamming modularity term is normalised by `1/(4m)` over `i<j`, like the one-hot and binary encodings, so `lambda_penalty` values are much smaller than they were in the archived notebooks (≈0.05 rather than 5–30 for the clique-chain examples).
