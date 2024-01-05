import unittest

from quantum_A.I._optimization import (COBYLA, VQE, Aer, EfficientSU2,
                                       NeuralNetworkClassifier,
                                       QuadraticProgram, TorchConnector,
                                       TwoLayerQNN, ansatz, feature_map,
                                       quantum_instance)


class TestQuantumAIOptimization(unittest.TestCase):
    def test_quadratic_program_creation(self):
        # Test case for creating a QuadraticProgram object
        problem = QuadraticProgram()
        self.assertIsInstance(problem, QuadraticProgram)

    def test_vqe_optimizer_creation(self):
        # Test case for creating a VQE optimizer object
        vqe_optimizer = VQE(optimizer=COBYLA(maxiter=100), quantum_instance=Aer.get_backend('qasm_simulator'))
        self.assertIsInstance(vqe_optimizer, VQE)

    def test_efficient_su2_creation(self):
        # Test case for creating an EfficientSU2 object
        feature_map = EfficientSU2(2, reps=1)
        self.assertIsInstance(feature_map, EfficientSU2)

    def test_two_layer_qnn_creation(self):
        # Test case for creating a TwoLayerQNN object
        ansatz = EfficientSU2(2, reps=1)
        qnn = TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance)
        self.assertIsInstance(qnn, TwoLayerQNN)

    def test_neural_network_classifier_creation(self):
        # Test case for creating a NeuralNetworkClassifier object
        qnn = TwoLayerQNN(2, feature_map, ansatz, quantum_instance=quantum_instance)
        nn_classifier = NeuralNetworkClassifier(qnn, TorchConnector(), epochs=10)
        self.assertIsInstance(nn_classifier, NeuralNetworkClassifier)

if __name__ == '__main__':
    unittest.main()
