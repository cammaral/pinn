import pennylane as qml


def make_ansatz(circuit_type="basic"):
    """
    Retorna uma função ansatz(weights, n_qubits) que aplica
    apenas a estrutura variacional do circuito.

    circuit_type:
        - "basic"
        - "strong"
    """

    if circuit_type == "basic":
        def ansatz(weights, n_qubits):
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return ansatz

    elif circuit_type == "strong":
        def ansatz(weights, n_qubits):
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return ansatz

    else:
        raise ValueError(f"circuit_type desconhecido: {circuit_type}")