import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# comparsion table
import pandas as pd

def comp_table(circuit): 
    df = pd.DataFrame(columns=['Original','Qiskit','PennyLane', 'tket'])
    df['Original'] = penny_specs(circuit)
    df['Qiskit'] = qiskit_specs(qiskit_optimizer(circuit))
    df['PennyLane'] = penny_specs(penny_optimizer(circuit))
    df['tket'] = tket_specs(tekt_optimizer(circuit))
    # df['QCC'] = summary_penny(my_loop(circuit))
    df.index = ['Gate count', 'Single-qubit gate count', 'Two-qubit gate count','Circuit depth']
    return df

def show_optimized_circuits(circuit):
    show_qc(qiskit_optimizer(circuit))
    show_qnode(penny_optimizer(circuit))
    show_tekt(tekt_optimizer(circuit))

# ===================================================================
# PennyLane
import pennylane as qml

def penny_optimizer(circuit):
    compiled = qml.compile(circuit)

    prev_gate_count = qml.specs(compiled)()['resources'].num_gates

    cnt = 0
    while cnt <=30:
        new_compiled = qml.compile(compiled)
        new_gate_count = qml.specs(new_compiled)()['resources'].num_gates

        if new_gate_count == prev_gate_count:
            compiled = new_compiled
            break

        compiled = new_compiled
        prev_gate_count = new_gate_count
        cnt += 1   
    return compiled

def penny_specs(circuit):
    obj = qml.specs(circuit)()['resources']
    return [obj.num_gates, obj.gate_sizes[1], obj.gate_sizes[2], obj.depth]

def show_qnode(circuit):
    qml.draw_mpl(circuit, style='pennylane')()
    plt.show()
# ===================================================================
# Qiskit
from qiskit import transpile as qiskit_transpiler
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import state_fidelity

def qiskit_optimizer(circuit):
    qc = qiskit_transpiler(to_qc(circuit),  basis_gates=["h","ry", "rz", "cx"], optimization_level=3)
    return qc

def show_qc(qc):
    qc.draw('mpl')
    plt.show()

def qiskit_specs(qc):
    counts = {'1q': 0, '2q': 0}
    for gate in qc.data:
        if len(gate.qubits) == 1:
            counts['1q'] += 1
        elif len(gate.qubits) == 2:  
            counts['2q'] += 1
    return [qc.size(), counts['1q'], counts['2q'], qc.depth()] 

def qc_fidelity(qc1, qc2):
    # Strip measurements before extracting the statevector
    qc_no_measure1 = qc1.remove_final_measurements(inplace=False)
    qc_no_measure2 = qc2.remove_final_measurements(inplace=False)

    # get statevectors
    st1 = Statevector.from_instruction(qc_no_measure1)
    st2 = Statevector.from_instruction(qc_no_measure2)

    return round(state_fidelity(st1, st2),3)    

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
        gate_info.append({
            "name": op.name,
            "wires": list(op.wires),
            "params": safe_params
        })
    return gate_info

def to_qiskit(qc, dict_elem):
    name = dict_elem['name']
    wire = dict_elem['wires']
    param = dict_elem['params']
    if name == 'Hadamard':
        qc.h(wire[0])
    if name == 'PauliX':
        qc.x(wire[0])
    if name == 'CNOT':
        qc.cx(wire[0],wire[1])
    if name == 'CZ':
        qc.cz(wire[0],wire[1])    
    if name == 'QubitUnitary':
        qc.append(UnitaryGate(param[0]),wire)
    if name == 'RY':
        qc.ry(param[0], wire[0])
    if name == 'U1':
        qc.p(param[0], wire[0])
    if name == 'U2':
        qc.u(np.pi/2, param[0], param[1], wire[0])

def wire_range(gate_dic):
    wire__list = [elem['wires'] for elem in gate_dic]
    flat = [item for sublist in wire__list for item in sublist]
    if min(flat)==max(flat):
        return [min(flat)]
    else:
        return [min(flat), max(flat)]
    
def to_qc(circuit):
    circuit_info = extract_info_from_qnode(circuit)
    sample_q_num = wire_range(circuit_info)[1] + 1
    qc = QuantumCircuit(sample_q_num)
    for dict_elem in circuit_info:
        to_qiskit(qc, dict_elem)
    return qc

# ===================================================================
# tket
from pytket.circuit import Circuit, QubitRegister, BitRegister, OpType
from pytket.passes import FullPeepholeOptimise, RebaseCustom
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.extensions.pennylane import pennylane_to_tk

def tekt_optimizer(circuit):
    tk_circ = to_tket(circuit)
    basis_set = {OpType.H, OpType.Ry, OpType.Rz, OpType.CX}
    cx_replacement = Circuit(2).CX(0, 1)
    FullPeepholeOptimise().apply(tk_circ)
    rebase_pass = RebaseCustom(basis_set, cx_replacement, tk1_to_rzryrz)
    rebase_pass.apply(tk_circ)
    return tk_circ

def show_tekt(tk_circ):
    tk_circ.replace_implicit_wire_swaps()
    qiskit_circ = tk_to_qiskit(tk_circ)
    qiskit_circ.draw("mpl")
    plt.show()

def tket_specs(tk_circ):
    total_gates = tk_circ.n_gates
    single_qubit_ops = [OpType.H, OpType.X, OpType.Ry, OpType.Rz]
    single_qubit_count = sum(tk_circ.n_gates_of_type(op) for op in single_qubit_ops)
    two_qubit_ops = [OpType.CX, OpType.CZ]
    two_qubit_count = sum(tk_circ.n_gates_of_type(op) for op in two_qubit_ops)
    depth = tk_circ.depth()
    return [total_gates, single_qubit_count, two_qubit_count, depth]

def tk1_to_rzryrz(a, b, c):
    circ = Circuit(1)
    circ.Rz(float(a), 0)
    circ.Ry(float(b), 0)
    circ.Rz(float(c), 0)
    return circ

def to_tket(circuit):
    qfunc = circuit.func
    with qml.tape.QuantumTape() as tape:
        qfunc()

    n_qubits = qml.specs(circuit)()['resources'].num_wires
    qreg = QubitRegister("q", n_qubits) 
    creg = BitRegister("c", 0)     
    wire_map = {w: i for i, w in enumerate(tape.wires)}
    tk_circ = pennylane_to_tk(tape.operations, wire_map, qreg, creg)
    return tk_circ
# ===================================================================