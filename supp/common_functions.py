import re
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from to_graph import To_graph

def qnode_info_obj(qnode=None, circuit_info=None):
    if qnode is not None:
        tg = To_graph(qnode=qnode)
    elif circuit_info is not None:
        tg = To_graph(circuit_info=circuit_info)
    return tg

def qnode_info(qnode=None, circuit_info=None):
    if qnode is not None:
        tg = To_graph(qnode=qnode)
    elif circuit_info is not None:
        tg = To_graph(circuit_info=circuit_info)
    G = tg.G
    circuit_info = tg.circuit_info
    return G, circuit_info

def extract_index(name: str) -> int:
            match = re.search(r'_(\d+)$', name)
            return int(match.group(1)) if match else None

def extract_info_from_qc(qc):
    circuit_info = []
    for instruction_obj in qc.data:
        op = instruction_obj.operation
        qargs = instruction_obj.qubits
        label = op.label if op.label else op.name.capitalize()
        wires = [qc.find_bit(q).index for q in qargs]
        params = [float(p) if hasattr(p, '__float__') else p for p in op.params]
        
        circuit_info.append({
            'label': label,
            'wires': wires,
            'params': params
        })
    return circuit_info
       
def info_to_qnode(circuit_info):
    max_wire = max(w for op in circuit_info for w in op["wires"])
    dev = qml.device('default.qubit', wires=range(max_wire+1))
    def circuit():
        for gate in circuit_info:
            if gate:
                apply_gate(gate)
        return qml.state()
    return qml.QNode(circuit, dev)

def apply_gate(gate):
    name = gate['label']
    wires = gate['wires']
    params = gate['params']

    # map simple gates directly
    simple_gates = {
        'Hadamard': lambda: qml.Hadamard(wires=wires[0]),
        'PauliX':   lambda: qml.PauliX(wires=wires[0]),
        'CNOT':     lambda: qml.CNOT(wires=wires),
        'CZ':       lambda: qml.CZ(wires=wires),
    }

    if name in simple_gates:
        simple_gates[name]()
    elif name in ['RX','RY','RZ']:
        if params == [0.0]:
            return
        getattr(qml, name)(params[0], wires=wires[0])
    elif name == 'QubitUnitary':
        qml.QubitUnitary(np.array(params[0]), wires=wires)
    else:
        raise ValueError(f"Unsupported gate: {name}")
    
def show_circuit(circuit):
    qml.draw_mpl(circuit, style='pennylane')()
    plt.show()

def show_info(circuit_info):
    qnode = info_to_qnode(circuit_info)
    qml.draw_mpl(qnode, style='pennylane')()
    plt.show()

def graph_info(G):
    for node, attr in G.nodes(data=True):
        print(node, attr)

# for test
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import math
from collections import defaultdict
import random

def show_graph(G, pos):
    gate_colors = {
        "Hadamard": "#f03028",
        "PauliX": "#fdae24",
        "RX": "#c9eb96",
        "RY": "#10b287",
        "RZ": "#127057",
        "CZ": "#4b96f2",
        "CNOT": "#332ec6",
        "QubitUnitary": "#6c024f"
    }

    node_colors = []
    for n in G.nodes:
        gate_label = G.nodes[n].get('label', '')
        node_colors.append(gate_colors.get(gate_label, 'lightgray'))

    _, ax = plt.subplots(figsize=(10, 5))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20, ax=ax)

    edge_counts = defaultdict(int)
    for u, v, k, d in G.edges(keys=True, data=True):
        pair = (u, v)
        edge_counts[pair] += 1
        count = edge_counts[pair]
        rad = 0 if count == 1 else 0.15 * ((-1) ** count) * math.ceil(count / 2)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               connectionstyle=f'arc3,rad={rad}',
                               ax=ax,
                               arrows=True,
                               arrowsize=5,
                               width=1)

    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in gate_colors.items()]
    ax.legend(handles=legend_patches,
              loc='upper right',
              fontsize='small',
              frameon=True,
              bbox_to_anchor=(1.2, 1))
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def community_graph(G, pos, communities):
    def color_generator(n):
        random.seed(42)
        colors = []
        for _ in range(n):
            hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(hex_color)
        return colors

    palette = color_generator(len(communities))
    node_color_map = {}
    for i, community in enumerate(communities):
        color = palette[i % len(palette)]
        for node in community:
            node_color_map[node] = color

    node_colors = [node_color_map.get(node, '#999999') for node in G.nodes]

    # Step 4: Draw the graph
    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=50, font_size=8, edge_color='gray', ax=ax)
    plt.title('Community graph')
    plt.tight_layout()
    plt.show()
