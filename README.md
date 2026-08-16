# iSCALE

**Spatial Coupling-Aware State Space Modeling for Mutation-Induced Binding Affinity Prediction**

iSCALE is a research implementation for predicting mutation-induced changes in protein–RNA binding affinity. It combines protein sequence descriptors, implicit multiscale information about the binding partner, and a bidirectional state space model. A structure-aware auxiliary task is used during training to shape the learned representation.

This repository is being prepared as the code release accompanying the manuscript:

> *Predicting Protein–RNA Binding Affinity Changes via Spatial Coupling-Aware State Space Modeling*

The repository was originally named DualPRI and early internal code used the name DualSSD. The public model name is **iSCALE**. The Python alias `DualSSD` and the command-line value `dualssd` are retained only for backward compatibility.

![iSCALE framework](Framework.png)

## Release scope

This code release contains:

- the iSCALE model and training code;
- the modified state space operations required to expose chunk-level states;
- protein–RNA dataset preprocessing utilities;
- training and cross-validation entry points;
- the configuration used for the manuscript release.

Large datasets, checkpoints, molecular-dynamics trajectories, and figure source data are distributed separately through a research-data repository. See [docs/DATA.md](docs/DATA.md).

## Repository layout

```text
.
├── config.py                     # Shared paths and manuscript model configuration
├── configs/paper_config.yaml     # Human-readable manuscript configuration
├── main.py                       # Training and evaluation entry point
├── cross_val.py                  # Cross-validation entry point
├── trainer.py                    # Training utilities
├── model/iscale.py               # iSCALE model implementation
├── mamba/                        # Modified state space implementation
├── dataset_process/              # Dataset preprocessing utilities
├── dataset/                      # Small benchmark metadata tables
├── docs/DATA.md                  # Data access and provenance
├── docs/REPRODUCIBILITY.md       # Manuscript reproduction guide
└── scripts/check_release.py      # Dependency-free release sanity check
```

## Environment

The reference environment uses Linux, Python 3.10, PyTorch 2.0, and CUDA 11.7. GPU execution requires a compatible NVIDIA driver.

### Conda installation

```bash
conda env create -f environment.yml
conda activate iscale
```

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

The custom state space kernels require Triton and are intended for Linux GPU environments. Do not use the machine-specific wheel paths that appeared in early repository revisions.

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

The public dataset DOI and checksums will be added to [docs/DATA.md](docs/DATA.md) before the versioned release. Raw third-party structural and benchmark data are not duplicated in the code repository.

## Manuscript configuration

The release candidate uses the following central configuration:

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

The machine-readable values are defined in `config.PAPER_MODEL_CONFIG` and documented in [configs/paper_config.yaml](configs/paper_config.yaml). These values must remain synchronized with the released checkpoint.

## Quick checks

Run the dependency-free repository check:

```bash
python scripts/check_release.py
```

After downloading the processed dataset, start a training run with the manuscript model name:

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

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the relationship between datasets, configurations, checkpoints, and manuscript figures.

## Outputs

Training outputs are written to timestamped directories under `output/` unless another location is supplied. These directories can contain checkpoints, logs, predictions, and cached intermediate values and are intentionally excluded from version control.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The manuscript DOI and the archived software DOI will be added when available.

## License and third-party code

The project is released under the Apache License 2.0. Portions of the state space implementation are derived from the Apache-2.0-licensed [state-spaces/mamba](https://github.com/state-spaces/mamba) project and have been modified to return intermediate states. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Datasets and external software retain their original terms of use and citation requirements.

## Contact

For questions about the code or data release, contact Rui Chen at `chenrui3074@stu.ouc.edu.cn`.
