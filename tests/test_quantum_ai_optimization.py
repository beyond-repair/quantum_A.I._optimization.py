"""Tests for the quantum AI optimization demo."""

from __future__ import annotations

import numpy as np

from quantum_ai_optimization import (
    ACTION_SPACE,
    build_problem,
    build_training_data,
    run_demo,
    solve_with_numpy,
    solve_with_sampling_vqe,
    train_qnn_classifier,
)


def test_problem_is_feasible_binary_qp():
    problem = build_problem()
    assert problem.get_num_binary_vars() == 2
    assert problem.get_num_linear_constraints() == 1
    # Feasible set must be non-empty for binary vars with x+y >= 1.
    result = solve_with_numpy(problem)
    assert result.status.name == "SUCCESS"
    assert float(result.x.sum()) >= 1.0


def test_numpy_solver_finds_known_optimum():
    problem = build_problem()
    result = solve_with_numpy(problem)
    # For this objective the unconstrained binary min is at (1,1); still feasible.
    np.testing.assert_allclose(result.x, [1.0, 1.0])
    assert result.fval == -11.0


def test_sampling_vqe_matches_numpy_on_tiny_qp():
    problem = build_problem()
    exact = solve_with_numpy(problem)
    vqe = solve_with_sampling_vqe(problem, maxiter=80, seed=42)
    np.testing.assert_allclose(vqe.x, exact.x)
    assert abs(vqe.fval - exact.fval) < 1e-6


def test_training_data_has_both_classes():
    X, y = build_training_data(np.array([1.0, 1.0]), seed=0)
    assert X.shape[1] == 2
    assert set(np.unique(y)).issuperset({-1, 1})


def test_qnn_classifier_trains_and_predicts():
    X, y = build_training_data(np.array([1.0, 1.0]), seed=7)
    clf, preds = train_qnn_classifier(X, y, maxiter=30, seed=7)
    assert preds.shape[0] == X.shape[0]
    assert clf.score(X, y) >= 0.5


def test_run_demo_end_to_end():
    results = run_demo(use_vqe=True, qnn_maxiter=30, vqe_maxiter=60, seed=42)
    assert results["solver"] == "SamplingVQE"
    assert results["score"] >= 0.5
    assert len(results["action_space"]) == len(ACTION_SPACE)
    np.testing.assert_allclose(results["solution"], [1.0, 1.0])


def test_run_demo_numpy_fallback_path():
    results = run_demo(use_vqe=False, qnn_maxiter=20, seed=1)
    assert results["solver"] == "NumPyMinimumEigensolver"
    assert results["objective"] == -11.0
