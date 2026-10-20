"""Analyse a saved Hamming hardware run (the result.json written by scripts/run.py --solver hardware --submit).

Subcommands (from the project root):
    python scripts/analysis.py correct results/<run>/result.json
        Syndrome-correct the raw hardware bitstring and report what changed. No jobs.
    python scripts/analysis.py noise results/<run>/result.json
        Test the raw bitstring against a uniform-noise model. No jobs.
    python scripts/analysis.py repeat results/<run>/result.json --n-repeats 10
        Re-sample the optimised circuit n times on the same backend. Costs n jobs.
"""

import argparse
import json
import os
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from networkx.algorithms.community import modularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graphUtility import plotting  # noqa: E402
from graphUtility.graphs import graph_from_spec  # noqa: E402
from graphUtility.plotting import plot_communities  # noqa: E402
from quantumEncodings.hamming import (HAMMING_CODES, assignments_from_data_bits,  # noqa: E402
                                      build_hamming_qaoa_circuit, compute_syndrome,
                                      correct_hamming_errors)
from quantumUtility.hardware import DEFAULT_ENV_PATH, connect_ibm_service, sample_output_hardware  # noqa: E402


def canonical_partition(assignment, N):
    """Relabel-invariant form of a community assignment.

    Community *labels* are gauge: {0,0,0,1,1,1} and {1,1,1,0,0,0} are the same
    partition.  Comparing raw label tuples would count a label swap as a
    different answer, so both are reduced to a sorted tuple of node-groups.

    Args:
        assignment (list[int] | dict): Community label of node i.
        N (int): Number of nodes.

    Returns:
        tuple: Sorted tuple of sorted node-index tuples.
    """
    groups = {}
    for n in range(N):
        groups.setdefault(assignment[n], []).append(n)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


def chance_baseline(partition, N, K):
    """Probability a uniformly random assignment reproduces `partition`.

    A random assignment matches iff each of the g groups receives a distinct
    label, so there are K*(K-1)*...*(K-g+1) favourable labelings out of K**N.

    Args:
        partition (tuple): Output of canonical_partition.
        N (int): Number of nodes.
        K (int): Number of communities.

    Returns:
        float: Probability in [0, 1].
    """
    g = len(partition)
    if g > K:
        return 0.0
    favourable = 1
    for i in range(g):
        favourable *= (K - i)
    return favourable / (K ** N)


def run_syndrome_correction_on_result(sampled_answer, G, N, K, code, show_graph=True, label="hardware"):
    """Apply Hamming syndrome correction to a sampled bitstring and report what changed.

    The hardware run returns the bitstring straight off the QPU, so correction
    is done here as a separate, inspectable step.  Keeping the two apart is
    what makes the code's contribution measurable: the interesting quantity is
    not the corrected string on its own but the *difference* between raw and
    corrected.

    What it reports
    ---------------
    - Per-node syndrome and correction status (via correct_hamming_errors).
    - How many of the N codewords were already valid.
    - Which nodes had a bit flipped back, and at which position.
    - Whether correction changed the community assignment.

    Idempotence
    -----------
    Correction maps every block onto a valid codeword, so a second pass finds
    all syndromes zero and changes nothing.

    Limits of Hamming(7,4)
    ----------------------
    Minimum distance 3, so it corrects any single-bit error but cannot detect a
    double-bit one: two flips carry a block onto a *different valid* codeword,
    which decoding silently accepts.  Hamming(8,4) can at least flag these.

    Args:
        sampled_answer (list of str | list of int | str): Raw bitstring, index j = logical qubit j.
        G (networkx.Graph): The problem graph, for re-colouring.
        N (int): Number of graph nodes.
        K (int): Number of communities.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        show_graph (bool): Redraw the graph from the corrected assignment.
        label (str): Name used in printed output, e.g. "hardware" or "simulator".

    Returns:
        dict: keys 'raw', 'corrected', 'assignment_raw', 'assignment_corrected',
            'n_valid', 'n_corrected', 'flipped' and 'assignment_changed'.
    """
    raw = [str(int(b)) for b in sampled_answer]
    L = code.length
    expected = N * L
    if len(raw) != expected:
        raise ValueError(f"bitstring has {len(raw)} bits, expected N*L = {N}*{L} = {expected}")

    # ── 1. State before correction ────────────────────────────────────────────
    syndromes_before = [compute_syndrome(raw[i*L:(i+1)*L], code.parity_checks) for i in range(N)]
    n_valid = sum(1 for s in syndromes_before if all(b == 0 for b in s))
    assignment_raw = assignments_from_data_bits(raw, N, K, L)

    print(f"\n{'='*72}")
    print(f"SYNDROME CORRECTION — {code.name}, {label} result")
    print(f"{'='*72}")
    print(f"Raw bitstring : {''.join(raw)}")
    print(f"Codewords already valid before correction: {n_valid}/{N} ({100*n_valid/N:.0f}%)")

    # ── 2. Correct (prints its own per-node table) ────────────────────────────
    corrected = correct_hamming_errors(raw, N, L, code.parity_checks)

    # ── 3. What actually changed ──────────────────────────────────────────────
    flipped = [(i, j % L) for i in range(N)
               for j in range(i*L, (i+1)*L) if raw[j] != corrected[j]]
    assignment_corrected = assignments_from_data_bits(corrected, N, K, L)
    changed = [i for i in range(N) if assignment_raw[i] != assignment_corrected[i]]

    print(f"\n{'-'*72}")
    print(f"Bits flipped back : {len(flipped)}"
          + ("  ->  " + ", ".join(f"node {i} bit {b}" for i, b in flipped) if flipped else ""))
    print(f"Nodes corrected   : {len({i for i, _ in flipped})}/{N}")

    print(f"\n{'Node':<6} {'Community (raw)':<17} {'Community (corrected)':<23} Changed")
    print("-" * 60)
    for i in range(N):
        mark = "  *" if i in changed else ""
        print(f"{i:<6} {assignment_raw[i]:<17} {assignment_corrected[i]:<23}{mark}")

    if changed:
        print(f"\nCorrection changed the assignment of {len(changed)} node(s): {changed}")
    else:
        print("\nCorrection did not change any community assignment.")

    # ── 4. Honest caveat about the code's reach ───────────────────────────────
    if L == 7:
        print(f"\nNote: Hamming(7,4) has d=3. Every non-zero syndrome is decoded as a\n"
              f"single-bit error, so a block carrying two flips is silently mapped onto\n"
              f"the wrong valid codeword. A clean correction report is therefore not\n"
              f"evidence of a correct answer when device fidelity is low.")

    if show_graph:
        plot_communities(G, assignment_corrected, f"Communities after {code.name} correction ({label})")

    return {"raw": raw,
            "corrected": corrected,
            "assignment_raw": assignment_raw,
            "assignment_corrected": assignment_corrected,
            "n_valid": n_valid,
            "n_corrected": len({i for i, _ in flipped}),
            "flipped": flipped,
            "assignment_changed": changed}


def repeat_final_sampling(record, n_repeats=10, apply_correction=True, env_path=DEFAULT_ENV_PATH):
    """Re-sample the optimised circuit n_repeats times and tabulate the answers.

    Does NOT re-run SPSA.  The logical circuit is rebuilt from the saved run
    configuration and bound to the saved optimised parameters, so each repeat
    costs one Sampler job on the same backend.

    What it reports
    ---------------
    - The decoded partition from every repeat, in relabel-invariant form.
    - How often the modal partition recurs (the reproducibility rate).
    - The uniform-random chance baseline for that partition, and a one-sided
      binomial tail giving the probability of seeing at least this many hits
      by luck.  A low tail is evidence of signal; a high one is not.

    Args:
        record (dict): A loaded result.json from a submitted hardware run.
        n_repeats (int): Number of sampling jobs to submit. 8-10 is enough to
            separate signal from a ~3% chance baseline.
        apply_correction (bool): Syndrome-correct each codeword before decoding.
        env_path (str): Path to the .env holding IBM_KEY and CRN.

    Returns:
        dict: Keys 'partitions', 'counts', 'modal', 'hits', 'n_repeats', 'baseline', 'p_value'.
    """
    config, result = record["config"], record["result"]
    code = HAMMING_CODES[config["encoding"]]
    N, K = record["N"], config["K"]

    G, _ = graph_from_spec(config["graph"])
    circuit, _, num_qubits = build_hamming_qaoa_circuit(G, N, K, config["P"], config["lam"],
                                                        code, config["mixer"])
    backend = connect_ibm_service(env_path).backend(result["backend"])

    print(f"Re-sampling {n_repeats}x on {backend.name} at the optimised parameters "
          f"({config['shots']} shots each, 1 job per repeat).\n")

    partitions = []
    for r in range(n_repeats):
        raw, _ = sample_output_hardware(circuit, backend, num_qubits,
                                        optimal_params=result["params"], shots=config["shots"])
        bits = [str(b) for b in raw]
        if apply_correction:
            bits = correct_hamming_errors(bits, N, code.length, code.parity_checks)

        part = canonical_partition(assignments_from_data_bits(bits, N, K, code.length), N)
        partitions.append(part)
        print(f"  repeat {r+1:2d}/{n_repeats}: {[list(g) for g in part]}")

    counts = Counter(partitions)
    modal, hits = counts.most_common(1)[0]
    baseline = chance_baseline(modal, N, K)
    # One-sided binomial tail: P(>= hits successes | p = baseline)
    p_value = sum(comb(n_repeats, j) * baseline**j * (1 - baseline)**(n_repeats - j)
                  for j in range(hits, n_repeats + 1))

    print("\n" + "=" * 66)
    print(f"{'partition':<44}{'count':>7}{'rate':>9}")
    print("-" * 66)
    for part, cnt in counts.most_common():
        print(f"{str([list(g) for g in part]):<44}{cnt:>7}{cnt/n_repeats:>8.0%}")
    print("-" * 66)
    print(f"modal partition        : {[list(g) for g in modal]}")
    print(f"reproducibility        : {hits}/{n_repeats} = {hits/n_repeats:.0%}")
    print(f"uniform-random baseline: {baseline:.4f} ({baseline:.2%} per repeat)")
    print(f"P(>= {hits} hits by chance) = {p_value:.2e}")
    print()
    if p_value < 0.01:
        print("=> Reproducible well beyond chance: the device is returning signal.")
    elif p_value < 0.05:
        print("=> Suggestive, but not decisive. Consider more repeats.")
    else:
        print("=> Consistent with chance. A single correct run was NOT evidence;\n"
              "   report the result as unresolved rather than as a success.")

    return {"partitions": partitions, "counts": counts, "modal": modal,
            "hits": hits, "n_repeats": n_repeats,
            "baseline": baseline, "p_value": p_value}


def test_noise_hypothesis(raw_bitstring, G, N, K, code, shots=4096, modal_prob=None,
                          reference_assignment=None, n_mc=20000, seed=0):
    """Check a hardware result against the predictions of a uniform-noise model.

    Under the global depolarising picture, a circuit with negligible fidelity
    returns samples drawn uniformly from the full computational basis.  That
    predicts the statistics of the observed bitstring on four independent axes:

      1. The modal bitstring occurs ~once, so its probability is ~1/shots.
      2. Each node's L bits land in the 16-codeword space with probability
         16/2^L, so the count of already-valid codewords is Binomial(N, 16/2^L).
      3. The decoded partition's modularity matches a random-assignment baseline.
      4. The number of distinct community labels used matches K(1-(1-1/K)^N).

    Predictions that hold do not prove the device was noisy -- they fail to
    refute it.  A partially-working circuit would break them: signal shows up
    as a repeated bitstring, an excess of valid codewords, or a modularity
    above the random band.

    Args:
        raw_bitstring (list[str] | list[int] | str): Bitstring BEFORE syndrome correction.
        G (networkx.Graph): The problem graph.
        N (int): Number of nodes.
        K (int): Number of communities.
        code (HammingCode): HAMMING_74 or HAMMING_84.
        shots (int): Shots used in the sampling job.
        modal_prob (float): Probability of the most likely bitstring. None skips prediction 1.
        reference_assignment (list[int]): A known-good partition to report the z-score of, or None.
        n_mc (int): Monte-Carlo draws for the random-modularity baseline.
        seed (int): RNG seed for the baseline.

    Returns:
        dict: Per-prediction observed/expected values and a 'consistent' flag.
    """
    bits = [str(int(b)) for b in raw_bitstring]
    L = code.length
    nodes = list(G.nodes())

    def to_partition(assign):
        groups = {}
        for i, c in enumerate(assign):
            groups.setdefault(c, set()).add(nodes[i])
        return list(groups.values())

    n_valid = sum(1 for i in range(N)
                  if all(s == 0 for s in compute_syndrome(bits[i*L:(i+1)*L], code.parity_checks)))
    assignment = assignments_from_data_bits(bits, N, K, L)

    # Baseline modularity for a uniformly random K-way assignment
    rng = np.random.default_rng(seed)
    mc = [modularity(G, to_partition(rng.integers(K, size=N))) for _ in range(n_mc)]
    mu, sd = float(np.mean(mc)), float(np.std(mc))
    q_obs = modularity(G, to_partition(assignment))

    p_valid = (2 ** (L - len(code.parity_checks))) / (2 ** L)   # 1/8 for (7,4)
    exp_valid = N * p_valid
    exp_used = K * (1 - (1 - 1.0 / K) ** N)
    used = len(set(assignment))

    print("NOISE-HYPOTHESIS TEST" + f"  ({N} nodes, K={K}, {shots} shots)")
    print("=" * 68)
    rows = []
    if modal_prob is not None:
        rows.append(("1. modal bitstring probability", f"~{1/shots:.2e}",
                     f"{modal_prob:.2e}", modal_prob <= 3.0 / shots))
    sd_valid = (N * p_valid * (1 - p_valid)) ** 0.5
    rows.append(("2. valid codewords pre-correction", f"{exp_valid:.2f} +/- {sd_valid:.2f}",
                 f"{n_valid}", abs(n_valid - exp_valid) <= 2.5 * sd_valid))
    z = (q_obs - mu) / sd if sd else float("inf")
    rows.append(("3. modularity of partition", f"{mu:+.4f} +/- {sd:.4f}",
                 f"{q_obs:+.4f} (z={z:+.2f})", abs(z) <= 2.0))
    rows.append(("4. distinct communities used", f"{exp_used:.2f}",
                 f"{used}", abs(used - exp_used) <= 2.5))

    print(f"{'prediction':<36}{'expected':>18}{'observed':>22}")
    print("-" * 76)
    for name, exp, obs, ok in rows:
        print(f"{name:<36}{exp:>18}{obs:>22}   {'MATCH' if ok else 'DEVIATES'}")
    print("-" * 76)

    consistent = all(r[3] for r in rows)
    if consistent:
        print("\nAll predictions hold: the output is statistically indistinguishable")
        print("from a uniform sample. The noise hypothesis is confirmed, and no")
        print("further repeats are needed to characterise this run.")
    else:
        print("\nAt least one prediction deviates -- the output carries structure a")
        print("uniform-noise model does not explain. Investigate before concluding.")
    if reference_assignment is not None and sd:
        print(f"\nFor reference, the true partition would give z = "
              f"{(modularity(G, to_partition(reference_assignment)) - mu) / sd:+.1f}")

    return {"n_valid": n_valid, "expected_valid": exp_valid,
            "q_observed": q_obs, "q_random_mean": mu, "q_random_sd": sd,
            "z": z, "used": used, "expected_used": exp_used,
            "consistent": consistent}


def clique_chain_reference(graph_spec, N):
    """The planted partition of a clique-chain graph (node n is in clique n // x), else None."""
    name, _, arg = graph_spec.partition(":")
    if name != "clique-chain":
        return None
    clique_size = int(arg.lower().split("x")[0])
    return [n // clique_size for n in range(N)]


def load_hardware_record(path):
    """Loads a result.json and checks it holds a submitted Hamming hardware run."""
    with open(path) as f:
        record = json.load(f)
    if record["config"]["solver"] != "hardware" or "raw_bitstring" not in (record["result"] or {}):
        raise SystemExit(f"{path} is not a submitted hardware run (dry runs hold no bitstring).")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["correct", "noise", "repeat"])
    parser.add_argument("result", help="path to a hardware result.json")
    parser.add_argument("--n-repeats", type=int, default=10, help="repeat: sampling jobs to submit")
    parser.add_argument("--no-correction", action="store_true", help="repeat: decode without syndrome correction")
    parser.add_argument("--env-path", default=DEFAULT_ENV_PATH, help="repeat: path to the .env")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()

    record = load_hardware_record(args.result)
    plotting.configure(save_dir=os.path.join(os.path.dirname(os.path.abspath(args.result)),
                                             f"analysis_{args.command}"),
                       show=args.show)
    config, result = record["config"], record["result"]
    code = HAMMING_CODES[config["encoding"]]
    N, K = record["N"], config["K"]
    G, _ = graph_from_spec(config["graph"])

    if args.command == "correct":
        run_syndrome_correction_on_result(result["raw_bitstring"], G, N, K, code)
    elif args.command == "noise":
        test_noise_hypothesis(result["raw_bitstring"], G, N, K, code, shots=config["shots"],
                              modal_prob=result["modal_probability"],
                              reference_assignment=clique_chain_reference(config["graph"], N))
    else:
        repeat_final_sampling(record, args.n_repeats, apply_correction=not args.no_correction,
                              env_path=args.env_path)


if __name__ == "__main__":
    main()
