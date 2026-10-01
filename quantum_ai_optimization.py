"""Quantum AI optimization demo: QuadraticProgram + SamplingVQE/NumPy + EstimatorQNN.

Preserves the original project intent (binary QP solved via a minimum-eigen
optimizer, then a small quantum neural network classifier trained on a
solution-derived dataset) while using modern Qiskit community packages.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_algorithms import NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.circuit.library import qnn_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA as QML_COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

# Historical sketch artifact: action labels from the original AI-agent framing.
ACTION_SPACE = [
    "x += 1",
    "x -= 1",
    "y += 1",
    "y -= 1",
]


def build_problem() -> QuadraticProgram:
    """Build a feasible two-variable binary quadratic program.

    Original sketch used ``x + y >= 5`` with binary vars (infeasible; max sum is 2).
    Constraint is ``x + y >= 1`` so the demo has a non-empty feasible set while
    keeping the same objective structure.
    """
    problem = QuadraticProgram(name="quantum_ai_binary_qp")
    problem.binary_var("x")
    problem.binary_var("y")
    problem.minimize(
        linear={"x": -6, "y": -8},
        quadratic={("x", "x"): 2, ("y", "y"): 2, ("x", "y"): -1},
    )
    problem.linear_constraint(
        linear={"x": 1, "y": 1},
        sense=">=",
        rhs=1,
        name="at_least_one",
    )
    return problem


def solve_with_numpy(problem: QuadraticProgram):
    """Exact classical minimum-eigen solve (reliable reference path)."""
    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    return solver.solve(problem)


def solve_with_sampling_vqe(
    problem: QuadraticProgram,
    *,
    maxiter: int = 80,
    seed: int = 42,
):
    """Variational quantum path via SamplingVQE + StatevectorSampler."""
    qubo = QuadraticProgramToQubo().convert(problem)
    operator, _offset = qubo.to_ising()
    ansatz = real_amplitudes(num_qubits=operator.num_qubits, reps=1)
    sampler = StatevectorSampler(seed=seed)
    mes = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
    )
    solver = MinimumEigenOptimizer(mes)
    return solver.solve(problem)


def build_training_data(
    solution: np.ndarray,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Small labeled set: +1 near the QP optimum, -1 elsewhere.

    Features are continuous 2-d points; labels use {-1, +1} as required by
    EstimatorQNN-backed NeuralNetworkClassifier.
    """
    rng = np.random.default_rng(seed)
    optimum = np.asarray(solution, dtype=float)
    X_list = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        optimum,
    ]
    # Jittered samples around corners and the reported solution.
    for base in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], optimum.tolist()):
        X_list.append(np.clip(np.asarray(base) + rng.normal(0, 0.08, size=2), 0.0, 1.0))

    X = np.asarray(X_list, dtype=float)
    # Label by proximity to the solved optimum in L2 distance.
    dists = np.linalg.norm(X - optimum.reshape(1, -1), axis=1)
    y = np.where(dists <= 0.35, 1, -1).astype(int)
    # Guarantee both classes exist for a trainable binary problem.
    if len(np.unique(y)) < 2:
        y[0] = -1
        y[-1] = 1
    return X, y


def train_qnn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    maxiter: int = 40,
    seed: int = 42,
) -> Tuple[NeuralNetworkClassifier, np.ndarray]:
    """Train a minimal EstimatorQNN classifier and return model + predictions."""
    algorithm_globals.random_seed = seed
    qc, feature_params, weight_params = qnn_circuit(num_qubits=2)
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_params,
        weight_params=weight_params,
        estimator=StatevectorEstimator(),
    )
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=QML_COBYLA(maxiter=maxiter),
    )
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    return classifier, predictions


def run_demo(
    *,
    use_vqe: bool = True,
    qnn_maxiter: int = 40,
    vqe_maxiter: int = 80,
    seed: int = 42,
) -> dict:
    """Run the full demo and return structured results."""
    problem = build_problem()
    numpy_result = solve_with_numpy(problem)

    vqe_result = None
    if use_vqe:
        vqe_result = solve_with_sampling_vqe(
            problem, maxiter=vqe_maxiter, seed=seed
        )
        solution = np.asarray(vqe_result.x, dtype=float)
        objective = float(vqe_result.fval)
        solver_name = "SamplingVQE"
    else:
        solution = np.asarray(numpy_result.x, dtype=float)
        objective = float(numpy_result.fval)
        solver_name = "NumPyMinimumEigensolver"

    X, y = build_training_data(solution, seed=seed)
    classifier, predictions = train_qnn_classifier(
        X, y, maxiter=qnn_maxiter, seed=seed
    )
    score = float(classifier.score(X, y))

    return {
        "solver": solver_name,
        "solution": solution,
        "objective": objective,
        "numpy_solution": np.asarray(numpy_result.x, dtype=float),
        "numpy_objective": float(numpy_result.fval),
        "X": X,
        "y": y,
        "predictions": predictions,
        "score": score,
        "action_space": ACTION_SPACE,
    }


def main() -> None:
    print("Quantum A.I. Optimization demo")
    print("(archive-queue sketch; not a quantum-advantage claim)\n")

    results = run_demo(use_vqe=True)

    print(f"Solver: {results['solver']}")
    x, y = results["solution"]
    print(f"Solution: x = {x}, y = {y}")
    print(f"Objective value: {results['objective']}")
    nx, ny = results["numpy_solution"]
    print(f"NumPy reference: x = {nx}, y = {ny}, fval = {results['numpy_objective']}")

    print(f"\nHistorical action-space sketch labels: {results['action_space']}")
    print(f"QNN training score (in-sample): {results['score']:.3f}")
    print("QNN predictions:")
    print(results["predictions"])

    print(
        f"\nThe optimal solution is x = {x} and y = {y}\n"
        f"The minimum objective value is {results['objective']}"
    )


if __name__ == "__main__":
    main()
