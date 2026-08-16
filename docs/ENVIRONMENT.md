# Environment

## Reference versions

| Component | Version |
|---|---:|
| Operating system | Linux |
| Python | 3.10.16 |
| PyTorch | 2.0.0+cu117 |
| torchvision | 0.15.0+cu117 |
| torchaudio | 2.0.0+cu117 |
| PyTorch CUDA runtime | 11.7 |
| CUDA NVCC | 11.8.89 |
| Triton | 2.2.0 |
| PyTorch Geometric | 2.6.1 |
| torch-scatter | 2.1.1+pt20cu117 |
| causal-conv1d | 1.4.0 |
| mamba-ssm | 2.2.2 |

The original causal-conv1d and mamba-ssm wheels were built for PyTorch 2.0
and labelled `cu118`. The repository uses its vendored state space kernels for
the core iSCALE path.

## Core installation

```bash
conda env create -f environment.yml
conda activate iscale
```

For a manual PyTorch Geometric installation:

```bash
pip install torch-scatter==2.1.1 \
  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Additional packages used in related workflows

| Workflow | Packages used in the working environment |
|---|---|
| Historical Mamba fast paths | causal-conv1d 1.4.0; mamba-ssm 2.2.2 |
| Optional sequence features | fair-esm 2.0.0; transformers 4.47.1 |
| Structural and MD analysis | MDAnalysis 2.9.0; OpenMM 8.2.0 |
| Additional PyG operators | torch-sparse 0.6.17; torch-cluster 1.6.1; torch-spline-conv 1.2.2 |

Install the historical Mamba packages after PyTorch when needed:

```bash
pip install --no-build-isolation causal-conv1d==1.4.0 mamba-ssm==2.2.2
```

## Version check

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
python -c "import torch_geometric, torch_scatter, triton; print(torch_geometric.__version__, torch_scatter.__version__, triton.__version__)"
```
