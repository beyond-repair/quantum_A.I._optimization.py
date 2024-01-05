import pytest
from quantum_A.I._optimization import (Aer, EfficientSU2,
                                       NeuralNetworkClassifier, TorchConnector,
                                       TwoLayerQNN, X, predictions, y)


@pytest.fixture
def quantum_instance():
    return Aer.get_backend('qasm_simulator')

@pytest.fixture
def qnn(quantum_instance):
    feature_map = EfficientSU2(2, reps=1)
    ansatz = EfficientSU2(2, reps=1)
    return NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)

@pytest.fixture
def test_data():
    X = np.array([[1, 1]])
    y = np.array([0])
    return X, y

def test_qnn_classifier(qnn, test_data):
    X, y = test_data
    assert len(X) == 1
    assert len(y) == 1
    assert X.shape[1] == 2
    assert predictions.shape == (1, 1)
