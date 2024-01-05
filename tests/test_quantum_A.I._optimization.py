import numpy as np
import pytest
from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.connectors import TorchConnector
from quantum_A.I._optimization import (MinimumEigenOptimizer, QuadraticProgram,
                                       TwoLayerQNN)


def test_optimization_problem_definition():
    problem = QuadraticProgram()
    problem.binary_var('x')
    problem.binary_var('y')
    problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
    problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)

    assert problem.variables == ['x', 'y']
    assert problem.objective.linear.to_dict() == {'x': -6, 'y': -8}
    assert problem.objective.quadratic.to_dict() == {('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1}
    assert problem.linear_constraints[0].linear.to_dict() == {'x': 1, 'y': 1}
    assert problem.linear_constraints[0].sense == '>='
    assert problem.linear_constraints[0].rhs == 5

def test_vqe_solver():
    problem = QuadraticProgram()
    problem.binary_var('x')
    problem.binary_var('y')
    problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
    problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)

    vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
    vqe = MinimumEigenOptimizer(vqe_optimizer)
    vqe_result = vqe.solve(problem)
    vqe_solution = vqe_result.x
    vqe_obj_value = vqe_result.fval

    assert len(vqe_solution) == 2
    assert isinstance(vqe_solution[0], int)
    assert isinstance(vqe_solution[1], int)
    assert isinstance(vqe_obj_value, float)

def test_quantum_neural_network_classifier():
    problem = QuadraticProgram()
    problem.binary_var('x')
    problem.binary_var('y')
    problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
    problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)

    vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
    vqe = MinimumEigenOptimizer(vqe_optimizer)
    vqe_result = vqe.solve(problem)
    vqe_solution = vqe_result.x

    quantum_instance = Aer.get_backend('qasm_simulator')
    feature_map = EfficientSU2(2, reps=1)
    ansatz = EfficientSU2(2, reps=1)
    qnn = NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)

    X = np.array([[vqe_solution[0], vqe_solution[1]]])
    y = np.array([0])

    qnn.fit(X, y)

    predictions = qnn.predict(X)

    assert len(predictions) == 1
    assert isinstance(predictions[0], int)

if __name__ == '__main__':
    pytest.main()
