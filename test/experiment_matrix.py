"""Run the experiment matrix (encoding x solver x K  against  graphs) and report a status per cell.

Every cell runs scripts/run.py in its own subprocess with a timeout, then
scores the saved result.json:

    ✅  ran, every node has a valid community, and the partition's modularity is
        at least QUALITY (90%) of a greedy-modularity reference with <= K communities
    ⚠️  ran, but the partition is poor, has invalid nodes, or no structure was found
    ❌  error, timeout, or more qubits than the simulator supports (not attempted)

Examples (from the project root):
    python test/experiment_matrix.py                              # everything
    python test/experiment_matrix.py --rows "Cl H74" "Q H74"      # some rows
    python test/experiment_matrix.py --cols Tri-2 Tri-3 --timeout 120

Writes report.md, report.json and one folder per cell (plots, result.json, run.log)
to test/results/<timestamp>/.
"""

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graphUtility.graphs import graph_from_spec  # noqa: E402

PLANTED = None  # K = number of planted communities of the graph (column default)

# (row label, encoding, solver, K or PLANTED, lambda or None for the encoding default)
ROWS = [
    ("Classical Binary Runs", "binary", "classical", PLANTED, 1.0),
    ("Quantum Binary Runs", "binary", "qaoa", PLANTED, 1.0),
    ("Cl 1-hot 2", "onehot", "classical", 2, 30.0),
    ("Q 1-hot 2", "onehot", "qaoa", 2, 1.0),
    ("Cl 1-hot 4", "onehot", "classical", 4, 30.0),
    ("Q 1-hot 4", "onehot", "qaoa", 4, 1.0),
    ("Cl 1-hot 8", "onehot", "classical", 8, 30.0),
    ("Q 1-hot 8", "onehot", "qaoa", 8, 1.0),
    ("Cl 1-hot 16", "onehot", "classical", 16, 30.0),
    ("Q 1-hot 16", "onehot", "qaoa", 16, 1.0),
    ("Cl H74", "hamming74", "classical", PLANTED, None),
    ("Q H74", "hamming74", "qaoa", PLANTED, None),
    ("Cl H84", "hamming84", "classical", PLANTED, None),
    ("Q H84", "hamming84", "qaoa", PLANTED, None),
]

# (column label, graph spec, planted number of communities)
COLUMNS = [
    ("Tri-2", "clique-chain:3x2", 2),
    ("Tri-3", "clique-chain:3x3", 3),
    ("Tri-4", "clique-chain:3x4", 4),
    ("Tri-8", "clique-chain:3x8", 8),
    ("Tri-16", "clique-chain:3x16", 16),
    ("Les Miserables", "les-miserables", 2),
    ("Karate Club", "karate", 2),
    ("Davis Southern Women", "davis", 2),
]

QUALITY = 0.9
ICONS = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

# Largest circuit each QAOA pipeline's simulator accepts (AerSimulator.num_qubits):
# one-hot uses statevector, binary "automatic" (statevector-sized), Hamming MPS.
SIMULATOR_QUBIT_LIMIT = {"onehot": 29, "binary": 29, "hamming74": 63, "hamming84": 63}


def qubits_needed(encoding, N, K):
    """Logical qubits the QAOA circuit for this cell needs."""
    K = min(K, N)
    if encoding == "onehot":
        return N * K
    if encoding == "binary":
        return N * math.ceil(math.log2(max(K, 2)))
    return N * (7 if encoding == "hamming74" else 8)


def partition_modularity(G, assignments):
    """Weighted modularity of a positional assignment; invalid (-1) nodes become singletons."""
    nodes = list(G.nodes())
    groups = {}
    for i, c in enumerate(assignments):
        groups.setdefault(c if c >= 0 else ("invalid", i), set()).add(nodes[i])
    return modularity(G, list(groups.values()), weight="weight")


def reference_modularity(G, K):
    """Modularity of a greedy partition with at most K communities."""
    communities = greedy_modularity_communities(G, weight="weight", best_n=min(K, G.number_of_nodes()))
    return modularity(G, communities, weight="weight")


def score(G, K, result):
    """Classifies a finished run as pass/warn from its saved result."""
    if result is None:
        return "warn", "spectral stop: no community structure found", None
    assignments = result["assignments"]
    q = partition_modularity(G, assignments)
    q_ref = reference_modularity(G, K)
    invalid = sum(1 for c in assignments if c < 0)
    detail = f"Q={q:.3f} vs greedy ref {q_ref:.3f}"
    if invalid:
        return "warn", f"{invalid} node(s) without a valid community; {detail}", q
    if q_ref > 0 and q >= QUALITY * q_ref:
        return "pass", detail, q
    return "warn", f"low modularity: {detail}", q


def run_cell(row, column, out_root, timeout):
    """Runs one experiment in a subprocess and returns its scored record."""
    label, encoding, solver, K, lam = row
    col_label, spec, planted = column
    K = planted if K is PLANTED else K

    with contextlib.redirect_stdout(io.StringIO()):
        G, N = graph_from_spec(spec)
    record = {"row": label, "column": col_label, "encoding": encoding, "solver": solver,
              "graph": spec, "N": N, "K": K, "lambda": lam}

    if solver == "qaoa":
        needed, limit = qubits_needed(encoding, N, K), SIMULATOR_QUBIT_LIMIT[encoding]
        record["qubits"] = needed
        if needed > limit:
            record.update(status="fail", reason=f"needs {needed} qubits > simulator limit {limit}",
                          seconds=0.0, modularity=None)
            return record

    out_dir = out_root / f"{label}__{col_label}".replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run.py"),
           "--encoding", encoding, "--solver", solver, "--graph", spec,
           "-K", str(K), "-P", "1", "--out", str(out_dir)]
    if lam is not None:  # None: the encoding's default (0.1/N for Hamming)
        cmd += ["--lam", str(lam)]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")

    start = time.time()
    with open(out_dir / "run.log", "w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  cwd=PROJECT_ROOT, timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    record["seconds"] = round(time.time() - start, 1)

    if returncode is None:
        record.update(status="fail", reason=f"timeout after {timeout}s", modularity=None)
    elif returncode != 0:
        lines = (out_dir / "run.log").read_text().strip().splitlines()
        record.update(status="fail", reason=(lines[-1] if lines else f"exit code {returncode}")[:160],
                      modularity=None)
    else:
        result = json.loads((out_dir / "result.json").read_text())["result"]
        status, reason, q = score(G, K, result)
        record.update(status=status, reason=reason, modularity=q)
    return record


def write_report(records, rows, columns, out_root):
    """Writes report.json and report.md (status grid + per-cell details)."""
    (out_root / "report.json").write_text(json.dumps(records, indent=2))
    by_cell = {(r["row"], r["column"]): r for r in records}
    col_labels = [c[0] for c in columns]

    lines = ["# Experiment matrix", "",
             f"✅ ran, valid, modularity ≥ {QUALITY:.0%} of greedy reference · "
             f"⚠️ ran but poor/invalid · ❌ error, timeout or too many qubits", "",
             "| Experiments Done | " + " | ".join(col_labels) + " |",
             "|---|" + "---|" * len(col_labels)]
    for row in rows:
        cells = [ICONS[by_cell[(row[0], c)]["status"]] if (row[0], c) in by_cell else " "
                 for c in col_labels]
        lines.append(f"| {row[0]} | " + " | ".join(cells) + " |")

    lines += ["", "## Details", "", "| Experiment | Graph | K | Status | Time (s) | Reason |", "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['row']} | {r['column']} | {r['K']} | {ICONS[r['status']]} | "
                     f"{r['seconds']} | {r['reason'].replace('|', '/')} |")
    report = "\n".join(lines) + "\n"
    (out_root / "report.md").write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="*", help="row labels to run (default: all)")
    parser.add_argument("--cols", nargs="*", help="column labels to run (default: all)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per cell")
    parser.add_argument("--jobs", type=int, default=4, help="cells run in parallel")
    parser.add_argument("--out", default=None, help="output directory (default test/results/<timestamp>)")
    args = parser.parse_args()

    rows = [r for r in ROWS if not args.rows or r[0] in args.rows]
    columns = [c for c in COLUMNS if not args.cols or c[0] in args.cols]
    out_root = Path(args.out or PROJECT_ROOT / "test" / "results" / f"{datetime.now():%Y%m%d-%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [(row, column) for row in rows for column in columns]
    print(f"Running {len(cells)} cells ({args.jobs} in parallel, {args.timeout}s timeout) -> {out_root}")

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_cell, row, column, out_root, args.timeout) for row, column in cells]
        for future in futures:
            r = future.result()
            records.append(r)
            print(f"{ICONS[r['status']]}  {r['row']:<22} {r['column']:<21} {r['seconds']:>7}s  {r['reason']}",
                  flush=True)

    print("\n" + write_report(records, rows, columns, out_root))


if __name__ == "__main__":
    main()
