import numpy as np
import pytest
from quantum_A.I._optimization import X, predictions, y


def test_qnn_classifier():
    X_test = np.array([[1, 2]])  # Test data for X
    y_test = np.array([0])  # Test data for y

    assert len(X) == 1
    assert len(y) == 1
    assert X.shape[1] == 2
    assert predictions.shape == (1, 1)

def test_predictions():
    X_test = np.array([[1, 2]])  # Test data for X

    assert len(X) == 1
    assert predictions.shape == (1, 1)
