"""All plotting: graphs, community partitions, convergence curves and circuits.

Every figure goes through finish(), which shows and/or saves it according to
configure().  The default (show, don't save) matches notebook use; scripts
usually call configure(save_dir=..., show=False).
"""

import os
import re

import matplotlib.pyplot as plt
import networkx as nx

# Distinct colours, one per community index; unassigned nodes are drawn grey.
COMMUNITY_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080', '#ffffff', '#000000',
]
UNASSIGNED_COLOR = 'gray'

_config = {"save_dir": None, "show": True}
_saved_count = 0


def configure(save_dir=None, show=True):
    """Sets where figures go.

    Args:
        save_dir (str, optional): Directory to save every figure into as PNG. None disables saving.
        show (bool): Display figures (inline in Jupyter, a window in a script).
    """
    _config["save_dir"] = save_dir
    _config["show"] = show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


def finish(fig, name):
    """Saves and/or shows a finished figure, then closes it.

    Args:
        fig (matplotlib.figure.Figure): The figure.
        name (str): Human-readable name; also used to build the file name.
    """
    global _saved_count
    if _config["save_dir"]:
        _saved_count += 1
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        fig.savefig(os.path.join(_config["save_dir"], f"{_saved_count:02d}_{slug}.png"),
                    bbox_inches="tight")
    if _config["show"]:
        try:
            from IPython import get_ipython
            in_notebook = get_ipython() is not None
        except ImportError:
            in_notebook = False
        if in_notebook:
            # Qiskit closes its own figures under the inline backend, so
            # plt.show() would miss circuit drawings; display() always works.
            from IPython.display import display
            display(fig)
        else:
            plt.show()
    plt.close(fig)


def display_graph(graph_input, title="Graph Visualization", node_colors='skyblue', seed=700):
    """Displays a NetworkX graph.

    Args:
        graph_input (networkx.Graph): The graph to be displayed.
        title (str, optional): The title of the graph visualization. Defaults to "Graph Visualization".
        node_colors (str or list, optional): The color(s) for the nodes. Can be a single color string
                                            or a list of colors, one for each node. Defaults to 'skyblue'.
        seed (int, optional): Seed for the random number generator used in `spring_layout`
                              to ensure consistent layout. Defaults to 700.
    """
    fig = plt.figure(figsize=(8, 6))
    # 'seed' ensures reproducibility of the layout; 'k' sets the optimal distance between nodes.
    pos = nx.spring_layout(graph_input, seed=seed, k=0.3)
    nx.draw_networkx(graph_input, pos, with_labels=True, node_color=node_colors,
                     node_size=700, edge_color='gray', font_size=10)
    plt.title(title)
    plt.axis('off')
    finish(fig, title)


def community_colors(assignments):
    """Maps per-node community indices to colours; -1 (invalid assignment) is grey.

    Args:
        assignments (list[int]): Community index of the i-th node of G.

    Returns:
        list: One colour per node.
    """
    colors = []
    for node_idx, community_idx in enumerate(assignments):
        if community_idx < 0:
            colors.append(UNASSIGNED_COLOR)
        elif community_idx >= len(COMMUNITY_COLORS):
            print(f"Warning: Not enough colors for community {community_idx}. "
                  f"Using default color for node {node_idx}.")
            colors.append(UNASSIGNED_COLOR)
        else:
            colors.append(COMMUNITY_COLORS[community_idx])
    return colors


def plot_communities(G, assignments, title="Detected Communities"):
    """Draws G with each node coloured by its community.

    Args:
        G (networkx.Graph): The original problem graph.
        assignments (list[int]): Community index of the i-th node of list(G.nodes()).
        title (str): Plot title.
    """
    display_graph(G, title, community_colors(assignments), seed=42)


def plot_convergence(tracker, title, xlabel="Iteration", ylabel="Cost", step=False):
    """Plots an objective value over optimiser iterations.

    Args:
        tracker (list[float]): Values recorded during optimisation.
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        step (bool): Draw as a step plot (for best-so-far trackers).
    """
    fig = plt.figure(figsize=(10, 4))
    if step:
        plt.step(range(len(tracker)), tracker, where="post", color="red", linewidth=1.5)
    else:
        plt.plot(tracker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    finish(fig, title)


def draw_circuit_mpl(circuit, title="Circuit", fold=-1, max_ops=1500, max_px=65000):
    """Draws the quantum circuit using Matplotlib.

    Matplotlib refuses to render a figure larger than 2**16 = 65536 pixels in
    either direction, and an unfolded QAOA ansatz on a few dozen qubits blows
    straight past that (RoC-2 under Hamming(7,4) is 42 qubits and thousands of
    gates).  Folding only trades width for height, so an oversized circuit is
    reported and skipped instead.

    circuit.draw('mpl') itself never raises on an oversized circuit -- it just
    returns a Figure -- and the ValueError is thrown later by matplotlib's PNG
    renderer, so the size has to be checked on the built figure before it is
    shown or saved.

    Args:
        circuit (qiskit.circuit.QuantumCircuit): The quantum circuit to draw.
        title (str, optional): The title to display above the circuit. Defaults to "Circuit".
        fold (int, optional): Column width to wrap the drawing at; -1 draws the
            circuit on a single line. Defaults to -1.
        max_ops (int, optional): Skip circuits with more operations than this,
            before spending time building a figure. Defaults to 1500.
        max_px (int, optional): Skip figures larger than this in either
            direction. Defaults to 65000 (matplotlib's own cap is 65536).
    """
    # Cheap pre-check: bail out before building a huge figure at all.
    if circuit.size() > max_ops:
        print(f"[draw_circuit_mpl] Skipped '{title}': {circuit.num_qubits} qubits, "
              f"{circuit.size()} ops, depth {circuit.depth()} — over max_ops={max_ops}. "
              f"Raise max_ops to override.")
        return

    fig = circuit.draw('mpl', fold=fold)

    # Authoritative check: matplotlib cannot render beyond 65536 px per side.
    width_px, height_px = fig.get_size_inches() * fig.dpi
    if max(width_px, height_px) > max_px:
        print(f"[draw_circuit_mpl] Skipped '{title}': figure is "
              f"{width_px:.0f}x{height_px:.0f} px, over the {max_px} px limit "
              f"({circuit.num_qubits} qubits, {circuit.size()} ops, depth {circuit.depth()}).")
        plt.close(fig)   # don't leak the unrendered figure
        return

    fig.suptitle(title)
    finish(fig, title)
