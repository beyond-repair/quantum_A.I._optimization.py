import numpy as np
import pytest
from quantum_A.I._optimization import (COBYLA, VQE, Aer, EfficientSU2,
                                       MinimumEigenOptimizer,
                                       NeuralNetworkClassifier,
                                       QuadraticProgram, TorchConnector,
                                       TwoLayerQNN)


@pytest.fixture
def optimization_problem():
    problem = QuadraticProgram()
    problem.binary_var('x')
    problem.binary_var('y')
    problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
    problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)
    return problem

@pytest.fixture
def vqe_optimizer():
    return VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))

@pytest.fixture
def vqe_result(optimization_problem, vqe_optimizer):
    vqe = MinimumEigenOptimizer(vqe_optimizer)
    return vqe.solve(optimization_problem)

def test_vqe_solution(vqe_result):
    vqe_solution = vqe_result.x
    assert len(vqe_solution) == 2
    assert isinstance(vqe_solution[0], int)
    assert isinstance(vqe_solution[1], int)

def test_vqe_obj_value(vqe_result):
    vqe_obj_value = vqe_result.fval
    assert isinstance(vqe_obj_value, float)

def test_action_space():
    action_space = [
        'x += 1',
        'x -= 1',
        'y += 1',
        'y -= 1',
    ]
    assert len(action_space) == 4
    assert all(isinstance(action, str) for action in action_space)

def test_qnn_training(vqe_result):
    vqe_solution = vqe_result.x
    X = np.array([[vqe_solution[0], vqe_solution[1]]])
    y = np.array([0])
    quantum_instance = Aer.get_backend('qasm_simulator')
    feature_map = EfficientSU2(2, reps=1)
    ansatz = EfficientSU2(2, reps=1)
    qnn = NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)
    qnn.fit(X, y)
    assert qnn.is_trained

def test_qnn_predictions(vqe_result):
    vqe_solution = vqe_result.x
    X = np.array([[vqe_solution[0], vqe_solution[1]]])
    quantum_instance = Aer.get_backend('qasm_simulator')
    feature_map = EfficientSU2(2, reps=1)
    ansatz = EfficientSU2(2, reps=1)
    qnn = NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)
    qnn.predict(X)
    assert len(qnn.predictions) == 1
    assert isinstance(qnn.predictions[0], int)
