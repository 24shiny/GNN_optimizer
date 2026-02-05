import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
import math
import pennylane as qml
import numpy as np
import re
from common_functions import *

def extract_info_from_qnode(qnode):
    quantum_fn = qnode.func
    with qml.tape.QuantumTape() as tape:
        quantum_fn()
    gate_info = []
    for op in tape.operations:
        safe_params = []
        for p in op.parameters:
            try:
                safe_params.append(float(p))
            except (TypeError, ValueError):
                safe_params.append(np.array(p).tolist())

        # skip rotation gates if all params are exactly 0.0
        if op.name in ["RX", "RY", "RZ", "U1", "U2"]:
            if all(val == 0.0 for val in safe_params):
                continue

        gate_info.append({
            "label": op.name,
            "wires": list(op.wires),
            "params": safe_params
        })
    return gate_info

class To_graph:
    def __init__(self, qnode=None, circuit_info=None):
        if circuit_info is not None:
            self.circuit_info = circuit_info
        elif qnode is not None:
            self.circuit_info = extract_info_from_qnode(qnode)
        else:
            raise ValueError("Provide either a qnode or circuit_info")

        self.gate_colors = {
            "Hadamard": "#f03028",
            "PauliX": "#fdae24",
            "RX": "#c9eb96",
            "RY": "#10b287",
            "RZ": "#10b287",
            "CZ": "#4b96f2",
            "CNOT": "#332ec6",
            "U1": "#6c024f",
            "U2": "#6c024f",
            "QubitUnitary": "#6c024f"
        }
        self.legend_patches = [mpatches.Patch(color=color, label=label)
                               for label, color in self.gate_colors.items()]
        self.G = None
        self.pos = None
        self.node_colors = []
        self.circuit_to_graph()
    
    def circuit_to_graph(self):
        self.G = nx.MultiDiGraph()
        max_wire = max(w for gate in self.circuit_info for w in gate['wires']) if self.circuit_info else -1
        num_qubits = max_wire + 1 if max_wire >= 0 else 0
        qubit_timelines = {q: [] for q in range(num_qubits)}

        for i, gate in enumerate(self.circuit_info):
            gate_id = f"{gate['label']}_{i}"
            # No 'label' attribute
            self.G.add_node(gate_id, label=gate['label'], params=gate['params'], wires=gate['wires'])
            for wire in gate['wires']:
                qubit_timelines[wire].append(gate_id)

        for q, timeline in qubit_timelines.items():
            for i in range(len(timeline) - 1):
                src = timeline[i]
                tgt = timeline[i + 1]
                wire = q 
                self.G.add_edge(src, tgt, key=f"wire_{wire}", wire=wire)

        self.pos = {}
        x_spacing = 0.6
        y_spacing = 0.8

        for i, gate in enumerate(self.circuit_info):
            gate_id = f"{gate['label']}_{i}"
            avg_y = -sum(gate['wires']) / len(gate['wires']) * y_spacing if gate['wires'] else 0
            self.pos[gate_id] = ((i + 1) * x_spacing, avg_y)

        self.node_colors = []
        for n in self.G.nodes:
            gate_label = self.G.nodes[n]['label']
            self.node_colors.append(self.gate_colors.get(gate_label, 'lightgray'))

    def show_graph(self):
        _, ax = plt.subplots(figsize=(10, 5))
        nx.draw_networkx_nodes(self.G, self.pos,node_color=self.node_colors,node_size=20,ax=ax)
        edge_counts = defaultdict(int)
        for u, v, k, d in self.G.edges(keys=True, data=True):
            pair = (u, v)
            edge_counts[pair] += 1
            count = edge_counts[pair]
            if count == 1:
                rad = 0
            else:
                rad = 0.15 * ((-1) ** count) * math.ceil(count / 2) 
            nx.draw_networkx_edges(self.G, self.pos, edgelist=[(u, v)],
                                   connectionstyle=f'arc3,rad={rad}', 
                                   ax=ax, 
                                   arrows=True, 
                                   arrowsize=5,
                                   width=1)
        ax.legend(handles=self.legend_patches, 
                  loc='upper right', 
                  fontsize='small', 
                  frameon=True, 
                  bbox_to_anchor=(1.2, 1))
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def augment_hadams(self):
        def extract_numbers(name):
            return list(map(int, re.findall(r'\d+', name)))

        def relabel_nodes(G):
            sorted_nodes = sorted(G.nodes(), key=extract_numbers)
            mapping = {}
            for counter, old in enumerate(sorted_nodes):
                label = G.nodes[old].get("label", "Gate")
                new_name = f"{label}_{counter}"
                mapping[old] = new_name
            return nx.relabel_nodes(G, mapping, copy=False)

        gate_2q = [n for n, attr in self.G.nodes(data=True) if len(attr.get('wires')) == 2]

        to_add = []
        to_edges = []
        to_pos = {}

        for u, v in list(self.G.edges()):
            if (u in gate_2q) and (v in gate_2q):
                wires_u = set(self.G.nodes[u]['wires'])
                wires_v = set(self.G.nodes[v]['wires'])

                # inter = wires_u.intersection(wires_v)
                # if len(inter) == 1:
                #     ref_wire = list(inter)
                # elif wires_u == wires_v:
                #     succs = [n for n in self.G.successors(u) if n != v]
                #     if succs:
                #         not_ref_wire = set(self.G.nodes[succs[0]]['wires'])
                #         ref_wire = list(wires_u.difference(not_ref_wire))
                #     else: continue 

                if wires_u == wires_v:
                    succs = [n for n in self.G.successors(u) if n != v]
                    if succs:
                        not_ref_wire = set(self.G.nodes[succs[0]]['wires'])
                        ref_wire = list(wires_u.difference(not_ref_wire))
                    else: continue # two directly connected gates
                else:
                    inter = wires_u.intersection(wires_v)
                    ref_wire = list(inter)

                idx = extract_numbers(u)[0]
                new_node1 = f"Hadamard_{idx}_1"
                new_node2 = f"Hadamard_{idx}_2"

                to_add.append((new_node1, {"label": "Hadamard", "params": [], "wires": ref_wire}))
                to_add.append((new_node2, {"label": "Hadamard", "params": [], "wires": ref_wire}))

                self.G.remove_edge(u, v)
                to_edges.extend([(u, new_node1), (new_node1, new_node2), (new_node2, v)])

                x_u, y_u = self.pos[u]
                x_v, y_v = self.pos[v]
                to_pos[new_node1] = ((x_u + x_v) / 2, (y_u + y_v) / 2 + 0.05)
                to_pos[new_node2] = ((x_u + x_v) / 2, (y_u + y_v) / 2 - 0.05)

        # apply additions
        self.G.add_nodes_from(to_add)
        self.G.add_edges_from(to_edges)
        self.pos.update(to_pos)

        # relabel once at the end
        self.G = relabel_nodes(self.G)

        # rebuild circuit_info with names
        self.circuit_info = [attr for node, attr in sorted(self.G.nodes(data=True), key=lambda x: extract_numbers(x[0]))]
