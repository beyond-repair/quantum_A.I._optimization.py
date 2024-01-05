import unittest

import numpy as np
from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit_machine_learning.algorithms import (NeuralNetworkClassifier,
                                                TwoLayerQNN)
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class TestOptimizationProblem(unittest.TestCase):
    def test_linear_terms(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        self.assertEqual(problem.linear.to_dict(), {'x': -6, 'y': -8})

    def test_quadratic_terms(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        self.assertEqual(problem.quadratic.to_dict(), {('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})

    def test_linear_constraints(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)
        self.assertEqual(problem.linear_constraints.to_dict(), [{'linear': {'x': 1, 'y': 1}, 'sense': '>=', 'rhs': 5}])

class TestVQESolver(unittest.TestCase):
    def test_vqe_solution(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)
        vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
        vqe = MinimumEigenOptimizer(vqe_optimizer)
        vqe_result = vqe.solve(problem)
        vqe_solution = vqe_result.x
        self.assertEqual(len(vqe_solution), 2)

    def test_vqe_obj_value(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)
        vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
        vqe = MinimumEigenOptimizer(vqe_optimizer)
        vqe_result = vqe.solve(problem)
        vqe_obj_value = vqe_result.fval
        self.assertIsInstance(vqe_obj_value, float)

class TestAIAgent(unittest.TestCase):
    def test_ai_agent_training(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)
        vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
        vqe = MinimumEigenOptimizer(vqe_optimizer)
        vqe_result = vqe.solve(problem)
        vqe_solution = vqe_result.x
        X = np.array([[vqe_solution[0], vqe_solution[1]]])
        y = np.array([0])
        quantum_instance = Aer.get_backend('qasm_simulator')
        feature_map = EfficientSU2(2, reps=1)
        ansatz = EfficientSU2(2, reps=1)
        qnn = NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)
        qnn.fit(X, y)
        self.assertTrue(qnn.is_fitted)

    def test_ai_agent_predictions(self):
        problem = QuadraticProgram()
        problem.binary_var('x')
        problem.binary_var('y')
        problem.minimize(linear=[-6, -8], quadratic={('x', 'x'): 2, ('y', 'y'): 2, ('x', 'y'): -1})
        problem.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=5)
        vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
        vqe = MinimumEigenOptimizer(vqe_optimizer)
        vqe_result = vqe.solve(problem)
        vqe_solution = vqe_result.x
        X = np.array([[vqe_solution[0], vqe_solution[1]]])
        y = np.array([0])
        quantum_instance = Aer.get_backend('qasm_simulator')
        feature_map = EfficientSU2(2, reps=1)
        ansatz = EfficientSU2(2, reps=1)
        qnn = NeuralNetworkClassifier(TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance), TorchConnector(), epochs=10)
        qnn.fit(X, y)
        predictions = qnn.predict(X)
        self.assertEqual(predictions, y)

if __name__ == '__main__':
    unittest.main()
