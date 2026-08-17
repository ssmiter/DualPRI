# iSCALE

**Spatial Coupling-Aware State Space Modeling for Mutation-Induced Binding Affinity Prediction**

iSCALE is a research implementation for predicting mutation-induced changes in protein–RNA binding affinity. It combines protein sequence descriptors, implicit multiscale information about the binding partner, and a bidirectional state space model. A structure-aware auxiliary task is used during training to shape the learned representation.

![iSCALE framework](Framework.png)

## Overview

The repository contains:

- the iSCALE model and training code;
- the modified state space operations required to expose chunk-level states;
- protein–RNA dataset preprocessing utilities;
- training and cross-validation entry points;
- the model configuration used in the study.

## Repository layout

```text
.
├── config.py                     # Shared paths and model configuration
├── configs/paper_config.yaml     # Human-readable model configuration
├── main.py                       # Training and evaluation entry point
├── cross_val.py                  # Cross-validation entry point
├── trainer.py                    # Training utilities
├── model/iscale.py               # iSCALE model implementation
├── mamba/                        # Modified state space implementation
├── dataset_process/              # Dataset preprocessing utilities
├── dataset/                      # Small benchmark metadata tables
├── docs/DATA.md                  # Data format and preparation notes
├── docs/ENVIRONMENT.md           # Runtime, compiler, and optional dependencies
├── docs/REPRODUCIBILITY.md       # Reproducibility notes
└── scripts/check_release.py      # Dependency-free release sanity check
```

## Environment

The reference environment uses Linux, Python 3.10, PyTorch 2.0.0 with CUDA 11.7, and CUDA 11.8 NVCC for custom extensions. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the complete environment specification.

### Conda installation

```bash
conda env create -f environment.yml
conda activate iscale
```

The Conda specification records `pytorch-cuda=11.7` for the PyTorch runtime and `cuda-nvcc=11.8.89` for compiling custom extensions.

### Manual installation

Install the PyTorch build appropriate for your CUDA runtime first. For the reference CUDA 11.7 environment:

```bash
conda create -n iscale python=3.10 -y
conda activate iscale
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 \
  -c pytorch -c nvidia
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

The custom state space kernels require Triton and are intended for Linux GPU environments. The included kernels are used directly by iSCALE; see [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for optional dependencies.

## Data preparation

The training entry points expect a processed pickle file containing paired wild-type and mutant protein graphs, the binding-partner graph, mutation metadata, and target affinity changes.

By default, the code looks for:

```text
data/processed/protein_rna_dataset.pkl
```

You can instead provide an explicit path:

```bash
python main.py --data_path /path/to/protein_rna_dataset.pkl
```

See [docs/DATA.md](docs/DATA.md) for the expected data format and preparation notes.

## Model configuration

The study uses the following central configuration:

| Parameter | Value |
|---|---:|
| Hidden dimension | 64 |
| SSD layers | 3 |
| State dimension | 32 |
| Convolution kernel size | 4 |
| Head dimension | 16 |
| Chunk size | 32 |
| Auxiliary loss weight | 0.2 |
| Dropout | 0.1 |

The machine-readable values are defined in `config.PAPER_MODEL_CONFIG` and documented in [configs/paper_config.yaml](configs/paper_config.yaml).

## Quick checks

Run the dependency-free repository check:

```bash
python scripts/check_release.py
```

After preparing the processed dataset, start a training run with:

```bash
python main.py \
  --model iscale \
  --data_path data/processed/protein_rna_dataset.pkl \
  --feature_type 1 \
  --batch_size 16 \
  --learning_rate 0.0008 \
  --epochs 300 \
  --chunk_size 32 \
  --patience 40 \
  --output_dir output
```

The feature options accepted by the current data loader are:

- `0`: sequence and evolutionary descriptors only;
- `1`: add multiscale spatial distribution features;
- `2`: add the nearest-partner coupling intensity;
- `3`: add both spatial feature types.

For five-fold evaluation, use:

```bash
python cross_val.py \
  --model iscale \
  --data_path data/processed/protein_rna_dataset.pkl \
  --feature_type 1 \
  --k_folds 5 \
  --chunk_size 32 \
  --seed 42
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for additional evaluation notes.

## Outputs

Training outputs are written to timestamped directories under `output/` unless another location is supplied. These directories can contain checkpoints, logs, predictions, and cached intermediate values and are intentionally excluded from version control.

## License and third-party code

The project is released under the Apache License 2.0. Portions of the state space implementation are derived from the Apache-2.0-licensed [state-spaces/mamba](https://github.com/state-spaces/mamba) project and have been modified to return intermediate states. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

For questions, contact Rui Chen at `chenrui3074@stu.ouc.edu.cn`.
