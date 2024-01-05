import numpy as np
import pytest
from qiskit_machine_learning.algorithms import (NeuralNetworkClassifier,
                                                TwoLayerQNN)
from qiskit_machine_learning.connectors import TorchConnector
from quantum_A.I._optimization import (Aer, EfficientSU2, predictions,
                                       vqe_obj_value, vqe_solution, y)


def test_qnn_classifier():
    # Setup
    quantum_instance = Aer.get_backend('qasm_simulator')
    feature_map = EfficientSU2(2, reps=1)
    ansatz = EfficientSU2(2, reps=1)
    qnn = NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)

    # Generate predictions using the trained model
    X = np.array([[vqe_solution[0], vqe_solution[1]]])
    qnn.fit(X, y)
    predictions = qnn.predict(X)

    # Assertions
    assert len(X) == 1
    assert len(y) == 1
    assert X.shape[1] == 2
    assert predictions.shape == (1, 1)

    # Print predictions and final solution with objective value
    print("Predictions:")
    print(predictions)
    print(f'The optimal solution is x = {vqe_solution[0]} and y = {vqe_solution[1]}')
    print(f'The minimum objective value is {vqe_obj_value}')
