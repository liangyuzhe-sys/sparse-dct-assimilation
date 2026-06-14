# Sparse-DCT Bayesian Data Assimilation

Code for the report *A Nonparametric Approach to Sparse Bayesian Data Assimilation
on Frozen Foundation-Model Backgrounds*.

The idea is to treat data assimilation as Bayesian series estimation in a DCT
basis instead of modelling the background-error covariance B explicitly. There
are two stages:

- **Stage 1 (no training):** estimate the analysis increment in a low-frequency
  DCT subspace with an L1 penalty (FISTA), then read off an explicit Gaussian
  posterior for the uncertainty. See `src/aurora_da/methods/sparse_dct.py`.
- **Stage 2 (small CNN):** a lightweight network that regresses the residual left
  by Stage 1 under a heteroscedastic Gaussian NLL, giving a mean and a pointwise
  variance. See `scripts/train_cnn_residual.py`.

The frozen **Aurora** model only provides background fields; **ERA5** is used as
the truth in an OSSE (observing-system simulation experiment).

This is a trimmed copy of my working repo — only the parts the report actually
uses. The exploratory variants I tried (block-DCT, wavelet, etc.) and the
debug/probe scripts are not included.

## Requirements

Python >= 3.11. Stage 1 is pure `numpy`/`scipy` (`scipy.fft.idctn`,
`scipy.linalg` Cholesky, `scipy.spatial.cKDTree`); Stage 2 uses `torch`. Data
I/O uses `xarray`/`zarr`/`gcsfs`, and backgrounds come from the
`microsoft-aurora` package. Pinned versions are in `pyproject.toml` / `uv.lock`.

## Layout

```
configs/         experiment settings (dates, grid, variables)
src/aurora_da/   library code
scripts/         command-line entry points
tests/           pytest unit tests
figures/         the two result figures used in the report
results/         run logs with the numbers in the paper
```

Library modules:

| File | Role |
|------|------|
| `methods/sparse_dct.py` | Stage 1: DCT-subspace L1/FISTA + energy truncation + Gaussian posterior |
| `observations.py`       | sparse random sampling operator + variable extraction |
| `metrics.py`            | cosine-of-latitude-weighted RMSE / cov90 helpers |
| `distances.py`          | great-circle geometry (used by `sparse_dct`) |
| `data/schema.py`        | variable/level schema and snapshot naming |
| `data/era5_fetch.py`    | fetch ERA5 from the ARCO-ERA5 store |
| `data/aurora_run.py`    | load frozen Aurora and run inference for backgrounds |

Scripts:

| Script | What it does |
|--------|--------------|
| `prepare_era5.py`           | download ERA5 snapshots |
| `prepare_background.py`     | run frozen Aurora to make backgrounds |
| `compute_baseline_stats.py` | per-variable `bg_rmse` and `sigma_obs` |
| `eval_dct_2023.py`          | Stage 1 results (Table 1 z500/z200/msl; cov90) |
| `prepare_training_data.py`  | build CNN training pairs (2021-2022) |
| `train_cnn_residual.py`     | train the Stage 2 residual CNN for one variable |
| `eval_cnn_2023.py`          | DCT+CNN results (Table 1 t/u/v/q rows; cov90) |
| `figures.py`                | make `figures/imp_bar.pdf` and `figures/calibration.pdf` |
| `verify_env.py`             | quick environment check |

## Setup

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run python scripts/verify_env.py
```

## Reproducing the numbers

The data isn't committed (ERA5 and the Aurora outputs are large). Rebuild it with:

```bash
uv run python scripts/prepare_era5.py
uv run python scripts/prepare_background.py
uv run python scripts/compute_baseline_stats.py
```

Then evaluate (52 dates x 3 seeds, sparse fraction p = 0.5%):

```bash
# Stage 1
uv run python -u scripts/eval_dct_2023.py --target z500

# Stage 2 (needs training pairs + a trained checkpoint)
uv run python -u scripts/prepare_training_data.py
uv run python -u scripts/train_cnn_residual.py --target t500
uv run python -u scripts/eval_cnn_2023.py --target t500
```

The logs under `results/` are my own runs, e.g. `results/eval_dct_z500.log` ends
with `imp = +32.7% +/- 6.4%`, which is the z500 number in Table 1.

## Tests

```bash
uv run pytest -v
```

## Notes

- Data-root and training-pair paths are set in `configs/data.yaml` and at the top
  of a few scripts (`/root/autodl-fs/...`, `/root/train_pairs/`); change them for
  your machine.
- `pyproject.toml` still lists `pywavelets` (a leftover from the wavelet variant I
  dropped); harmless, kept so `uv.lock` stays valid.
