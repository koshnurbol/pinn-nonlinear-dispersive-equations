# Physics-Informed Neural Networks for Nonlinear Dispersive Equations

This repository contains the code accompanying the paper

> **Physics-Informed Neural Networks for Nonlinear Dispersive Equations**
> N. Koshkarbayev, Institute of Mathematics and Mathematical Modeling, Almaty, Kazakhstan.

It reproduces all numerical results, ablation studies, seed-robustness tables, and
figures reported in the paper for four canonical dispersive equations:

| Equation | Operator | Domain | Final time |
|----------|----------|--------|------------|
| Korteweg–de Vries (KdV) | `u_xxx` | [-10, 10] | 1.5 |
| Benjamin–Bona–Mahony (BBM) | `u_xxt` | [-20, 20] | 5.0 |
| Kawahara | `u_xxxxx` | [-20, 20] | 2.0 |
| Rosenau–KdV | `u_xxxxt` | [-20, 20] | 2.0 |

All experiments use a standard fully connected physical-space PINN with `tanh`
activations and automatic differentiation for every derivative. No Fourier
features, spectral layers, or equation-specific basis functions are used.

## Repository structure

```
.
├── src/
│   ├── pinn_kdv.py         # KdV experiment
│   ├── pinn_bbm.py         # BBM experiment
│   ├── pinn_kawahara.py    # Kawahara experiment (also saves the trained model)
│   └── pinn_rosenau.py     # Rosenau–KdV experiment
├── diagnostics/
│   ├── kawahara_error_spectrum.py        # Fourier error-spectrum figure
│   └── kawahara_phase_decomposition.py   # phase/spectral control experiment
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The code runs on CPU or GPU; a CUDA-capable GPU is recommended for the
fifth-order equations (Kawahara, Rosenau–KdV).

## Running the experiments

Each script in `src/` is self-contained. The training configuration is controlled
by three flags at the top of the file:

```python
SEED = 2026                       # random seed
USE_INPUT_NORMALIZATION = False   # normalize inputs to [-1, 1]
USE_ENERGY_LOSS = True            # add the invariant-preserving penalty
USE_LBFGS = True                  # apply L-BFGS refinement after Adam
```

To run a single experiment:

```bash
cd src
python pinn_kdv.py
```

Each run prints the metrics (final-time relative L2, space-time L2, PDE residual,
energy drift, training time), appends a row to a per-equation CSV file, and saves
the solution-profile, L2(t), and energy-drift figures.

### Reproducing the ablation tables

The ablation tables in the paper correspond to toggling the three flags. For
example, for KdV (Table 1):

| Case | `USE_INPUT_NORMALIZATION` | `USE_ENERGY_LOSS` | `USE_LBFGS` |
|------|---------------------------|-------------------|-------------|
| A0   | True  | True  | True  |
| A1   | False | True  | True  |
| A2   | True  | False | True  |
| A3   | True  | True  | False |
| A4   | False | False | True  |

The BBM, Kawahara, and Rosenau–KdV ablation tables follow the same pattern. The
configuration selected for each equation's seed-robustness study is:

| Equation | Selected config | Flags |
|----------|-----------------|-------|
| KdV          | A1 | no norm, energy, L-BFGS |
| BBM          | B2 | norm, no energy, L-BFGS |
| Kawahara     | K2 | norm, no energy, L-BFGS |
| Rosenau–KdV  | R0 | norm, energy, L-BFGS    |

### Reproducing the seed-robustness tables

Run the selected configuration with each of the three seeds used in the paper:

```python
SEED = 42      # then 123, then 2026
```

The reported values are the mean ± sample standard deviation over the three seeds.

## Reproducing the spectral-error analysis (Kawahara)

The Fourier error-spectrum figure and the phase-decomposition control are produced
from a trained Kawahara model. First train and save the model:

```bash
cd src
python pinn_kawahara.py          # writes kawahara_standard_pinn_model.pt
```

Then run the diagnostics from the same directory (so the `.pt` file is found):

```bash
python ../diagnostics/kawahara_error_spectrum.py        # Hann-windowed error spectrum
python ../diagnostics/kawahara_phase_decomposition.py   # phase-shift control
```

The first script reports the spectral centroids of the solution and of the error
and saves the figure used in the paper. The second confirms that the
high-wavenumber content of the error is intrinsic, not an artifact of a rigid
phase shift.

## Notes on reproducibility

- Random seeds are fixed for `torch` and `numpy`. Results on GPU may differ
  slightly from CPU due to non-deterministic CUDA kernels; the reported trends and
  order of magnitude are stable across hardware.
- The Rosenau–KdV script has a `FAST_MODE` flag. The default (`FAST_MODE = False`)
  reproduces the paper results; setting it to `True` runs a reduced-budget version
  for quick checks and will not match the published numbers exactly.
  
## Code availability

This repository contains the implementation used to generate the numerical
results, ablation studies, seed-robustness tables, solution-profile figures,
energy-drift diagnostics, and Kawahara spectral-error analysis reported in the
accompanying manuscript.

The repository is currently under preparation for public release. The archived
version corresponding to the manuscript will be deposited on Zenodo and cited by
DOI.

## License

This code is released under the MIT License. If you use this repository, please
cite the accompanying paper and the archived software release.

## Citation

Citation metadata are provided in `CITATION.cff`. After the manuscript is
accepted or a Zenodo DOI is assigned, this section will be updated with the final
paper citation and archived software DOI.
