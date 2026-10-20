"""Run one community-detection experiment and save its plots and result.json.

Examples (from the project root):
    python scripts/run.py --encoding onehot    --solver qaoa      --graph clique-chain:3x8 -K 2 -P 1 --lam 1
    python scripts/run.py --encoding onehot    --solver classical --graph davis            -K 2 --lam 30
    python scripts/run.py --encoding binary    --solver qaoa      --graph clique-chain:3x3 -K 3 -P 1
    python scripts/run.py --encoding hamming74 --solver classical --graph karate           -K 2
    python scripts/run.py --encoding hamming74 --solver hardware  --graph clique-chain:3x2 -K 2
    python scripts/run.py --encoding hamming74 --solver hardware  --graph clique-chain:3x2 -K 2 --submit

Hardware runs are dry runs (nothing submitted) unless --submit is given.
Hamming runs default to lambda = 0.1/N (see quantumEncodings.hamming.default_hamming_lambda).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classicalEncoding.binary import run_binary_classical  # noqa: E402
from classicalEncoding.hamming import run_hamming_classical  # noqa: E402
from classicalEncoding.onehot import run_onehot_classical  # noqa: E402
from graphUtility import plotting  # noqa: E402
from graphUtility.graphs import graph_from_spec  # noqa: E402
from quantumEncodings.binary import run_binary_qaoa  # noqa: E402
from quantumEncodings.hamming import HAMMING_CODES, run_hamming_hardware, run_hamming_qaoa  # noqa: E402
from quantumEncodings.onehot import run_onehot_qaoa  # noqa: E402
from quantumUtility.ansatz import INITIAL_BETA, INITIAL_GAMMA  # noqa: E402
from quantumUtility.hardware import DEFAULT_ENV_PATH  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoding", required=True, choices=["onehot", "binary", "hamming74", "hamming84"])
    parser.add_argument("--solver", required=True, choices=["qaoa", "classical", "hardware"],
                        help="qaoa = AerSimulator, classical = baseline, hardware = IBM QPU (Hamming only)")
    parser.add_argument("--graph", default="clique-chain:3x2",
                        help="clique-chain:XxY, initial, karate, les-miserables or davis")
    parser.add_argument("-K", type=int, required=True, help="number of communities")
    parser.add_argument("-P", type=int, default=1, help="number of QAOA layers")
    parser.add_argument("--lam", type=float, default=None,
                        help="penalty strength (lambda); default 0.1/N for Hamming, 1.0 otherwise")
    parser.add_argument("--gamma", type=float, default=INITIAL_GAMMA, help="initial γ for every layer")
    parser.add_argument("--beta", type=float, default=INITIAL_BETA, help="initial β for every layer")
    parser.add_argument("--mixer", choices=["generator", "x"], default=None,
                        help="Hamming mixer; default generator for (7,4), x for (8,4)")
    parser.add_argument("--seed", type=int, default=None, help="seed (classical solvers and SPSA)")

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument("--submit", action="store_true", help="actually submit jobs (default: dry run)")
    hardware.add_argument("--n-iterations", type=int, default=20, help="SPSA steps = jobs")
    hardware.add_argument("--shots", type=int, default=4096)
    hardware.add_argument("--env-path", default=DEFAULT_ENV_PATH, help="path to the .env with IBM_KEY and CRN")

    output = parser.add_argument_group("output")
    output.add_argument("--out", default=None, help="output directory (default results/<timestamp>_<encoding>_<solver>)")
    output.add_argument("--show", action="store_true", help="also display figures interactively")
    return parser.parse_args()


def run(args, G, N):
    is_hamming = args.encoding.startswith("hamming")
    if args.solver == "hardware" and not is_hamming:
        raise SystemExit("The hardware solver is only implemented for the Hamming encodings.")

    if args.encoding == "onehot":
        lam = 1.0 if args.lam is None else args.lam
        if args.solver == "qaoa":
            return run_onehot_qaoa(G, N, args.K, args.P, lam, args.gamma, args.beta)
        return run_onehot_classical(G, N, args.K, lam, seed=42 if args.seed is None else args.seed)

    if args.encoding == "binary":
        if args.solver == "qaoa":
            return run_binary_qaoa(G, N, args.K, args.P, args.gamma, args.beta)
        return run_binary_classical(G, N, args.K, seed=42 if args.seed is None else args.seed)

    code = HAMMING_CODES[args.encoding]
    if args.solver == "qaoa":
        return run_hamming_qaoa(G, N, args.K, args.P, args.lam, code, args.mixer, args.gamma, args.beta)
    if args.solver == "classical":
        return run_hamming_classical(G, N, args.K, args.lam, code, seed=42 if args.seed is None else args.seed)
    return run_hamming_hardware(G, N, args.K, args.P, args.lam, code, args.mixer,
                                n_iterations=args.n_iterations, shots=args.shots,
                                env_path=args.env_path, seed=123 if args.seed is None else args.seed,
                                dry_run=not args.submit, initial_gamma=args.gamma, initial_beta=args.beta)


def main():
    args = parse_args()
    out_dir = args.out or os.path.join(
        PROJECT_ROOT, "results", f"{datetime.now():%Y%m%d-%H%M%S}_{args.encoding}_{args.solver}")
    plotting.configure(save_dir=out_dir, show=args.show)

    G, N = graph_from_spec(args.graph)
    result = run(args, G, N)

    config = {key: value for key, value in vars(args).items() if key not in ("env_path", "out", "show")}
    record = {"config": config, "N": N, "result": result}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(record, f, indent=2, default=float)
    print(f"\nSaved plots and result.json to {out_dir}")


if __name__ == "__main__":
    main()
