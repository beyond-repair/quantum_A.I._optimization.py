<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║   ATOMIC DREAM LABS  ·  BEYOND-REPAIR                        ║
╚══════════════════════════════════════════════════════════════╝
```

# Quantum A.I. Optimization

### Qiskit sketch. Not a demonstrated quantum advantage.

[![Lifecycle](https://img.shields.io/badge/●_ARCHIVE-64748b?style=for-the-badge&labelColor=0f0f23)](https://github.com/beyond-repair/ADL-Governance)
[![Claim](https://img.shields.io/badge/Claim_0-22c55e?style=for-the-badge&labelColor=0f0f23)](https://github.com/beyond-repair/ADL-Governance/blob/main/docs/CLAIM_VALIDATION.md)
[![Governance](https://img.shields.io/badge/ADL--Governance-7c3aed?style=for-the-badge&labelColor=0f0f23)](https://github.com/beyond-repair/ADL-Governance)

```
LIFECYCLE   ARCHIVE QUEUE
CLAIM       0
NOT CLAIMED profit · live trading · product · quantum advantage
```

</div>

---
> **ARCHIVE QUEUE.** Historical demo sketch. No profit, deployment, product, or quantum-advantage claim.

## ▌ STATUS

Archive-queue under [ADL-Governance](https://github.com/beyond-repair/ADL-Governance). Runnable educational demo only. Do not use for live trading, deployment, or as a product.

---

## ▌ PRESERVED BODY

# Quantum Optimization with AI

This project demonstrates a small integration of quantum optimization and quantum machine learning with Qiskit community packages:

1. Define a feasible two-variable **binary QuadraticProgram**.
2. Solve it with **SamplingVQE** (variational path) and compare against **NumPyMinimumEigensolver**.
3. Train a minimal **EstimatorQNN** classifier on a tiny dataset derived from the solved optimum.

It is a Claim-0 archive sketch, not a production optimizer and not a claim of quantum advantage.

## Requirements

- Python 3.10+ (tested on 3.13 locally; CI matrix may use 3.9–3.11)
- See `requirements.txt` for pinned Qiskit packages

## Installation

```bash
git clone https://github.com/beyond-repair/quantum_A.I._optimization.py.git
cd quantum_A.I._optimization.py
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python quantum_A.I._optimization.py
# equivalent:
python -m quantum_ai_optimization
```

The script prints the SamplingVQE solution, the NumPy reference, and in-sample QNN predictions.

## Tests

```bash
pytest -q
```

## Project layout

| Path | Role |
| --- | --- |
| `quantum_A.I._optimization.py` | Original entrypoint name (thin shim) |
| `quantum_ai_optimization.py` | Importable demo logic |
| `tests/` | Pytest coverage for QP, VQE, and QNN paths |
| `requirements.txt` | Pinned dependencies |

## Notes on the repair

- Original constraint `x + y >= 5` with binary variables was infeasible (max sum is 2). It is now `x + y >= 1`.
- Deprecated `qiskit.Aer` / `qiskit.algorithms.VQE` / broken `TwoLayerQNN` imports were replaced with `SamplingVQE`, `NumPyMinimumEigensolver`, and `EstimatorQNN` + `NeuralNetworkClassifier`.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Qiskit](https://www.ibm.com/quantum/qiskit)
- [Qiskit Optimization](https://qiskit-community.github.io/qiskit-optimization/)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Qiskit Algorithms](https://qiskit-community.github.io/qiskit-algorithms/)

## Contact

For questions: williambrianware84@gmail.com

---

<div align="center">

**REWRITE · BUILD · TRANSCEND**

**William (Brian) Ware** · [Atomic Dream Labs](https://github.com/beyond-repair)  
Governing source: [ADL-Governance](https://github.com/beyond-repair/ADL-Governance) · [Claim levels 0–5](https://github.com/beyond-repair/ADL-Governance/blob/main/docs/CLAIM_VALIDATION.md)

</div>
