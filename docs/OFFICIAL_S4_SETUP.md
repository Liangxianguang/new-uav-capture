# Official S4 backend

The repository contains two different S4-labelled paths and keeps their
provenance explicit:

- `s4_diffusion` uses the dependency-free `s4_dplr_dense_reference` backend.
- `official_s4_diffusion` adapts the upstream `state-spaces/s4`
  `SSMKernelDPLR` implementation and records hashes for the upstream SSM,
  DPLR, and HiPPO source files in the checkpoint configuration.

The upstream source is an optional runtime dependency. It is not vendored into
this repository and it is not equivalent to the optional CUDA kernel. Without
the CUDA extension, the upstream adapter uses its own naive Cauchy/Vandermonde
fallback, which is correct for a small reproducibility smoke but is slower.

## Setup

Use the commit tested by this project:

```text
e757cef57d89e448c413de7325ed5601aceaac13
```

```powershell
git clone https://github.com/state-spaces/s4.git external\state-spaces-s4
git -C external\state-spaces-s4 checkout e757cef57d89e448c413de7325ed5601aceaac13
python -m pip install -r requirements-official-s4.txt
```

The project predictor can then be trained with:

```powershell
python scripts\train_prediction_models.py `
  --model official_s4_diffusion `
  --official-s4-root external\state-spaces-s4 `
  --train-dataset results\phase15_s4_branching_predictor_v3\train\dataset.npz `
  --validation-dataset results\phase15_s4_branching_predictor_v3\validation\dataset.npz `
  --output models\phase15_s4_v3_official_s4_action `
  --action-conditioning both
```

Evaluate the frozen model with the same source root:

```powershell
python scripts\evaluate_prediction_models.py `
  --checkpoint models\phase15_s4_v3_official_s4_action\checkpoint.pt `
  --training-output models\phase15_s4_v3_official_s4_action `
  --official-s4-root external\state-spaces-s4 `
  --validation-dataset results\phase15_s4_branching_predictor_v3\validation\dataset.npz `
  --locked-test-dataset results\phase15_s4_branching_predictor_v3\locked_test\dataset.npz `
  --output results\phase15_s4_v3_locked_test\official_s4_action
```

The checkpoint is not portable without the upstream source tree. The
evaluation command therefore requires an explicit source root, and refuses to
silently substitute the local dense reference implementation.
