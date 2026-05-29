import pennylane as qml

def ms_brickwall(params, n_qubits):
    depth = params.shape[0]
    for layer in range(depth):
        if layer % 2 == 0:
            rotation_type = ['x', 'y', 'z'][(layer // 2) % 3]
            for i in range(n_qubits):
                angle = params[layer, i]
                if rotation_type == 'x':
                    qml.RX(angle, wires=i)
                elif rotation_type == 'y':
                    qml.RY(angle, wires=i)
                else:
                    qml.RZ(angle, wires=i)
        else:
            offset = (layer // 2) % 2
            for i in range(offset, n_qubits - 1, 2):
                angle = params[layer, i]
                qml.IsingXX(angle, wires=[i, i + 1])
        qml.Barrier(wires=range(n_qubits))