# Reproducing the manuscript analyses

## Release pairing

Use the versioned iSCALE code together with the processed dataset and source
data archived at [10.5281/zenodo.21972360](https://doi.org/10.5281/zenodo.21972360).
The model configuration used in the study is recorded in:

- `config.PAPER_MODEL_CONFIG` for Python entry points;
- `configs/paper_config.yaml` for a human-readable summary.

## Core training command

```bash
python main.py \
  --model iscale \
  --data_path data/processed/protein_rna_dataset.pkl \
  --feature_type 1 \
  --batch_size 16 \
  --learning_rate 0.0008 \
  --weight_decay 0.00001 \
  --epochs 300 \
  --patience 40 \
  --chunk_size 32 \
  --seed 42
```

## Five-fold evaluation

```bash
python cross_val.py \
  --model iscale \
  --data_path data/processed/protein_rna_dataset.pkl \
  --feature_type 1 \
  --k_folds 5 \
  --chunk_size 32 \
  --seed 42
```

Model optimization is stochastic, so exact results can depend on the software
environment and hardware. The commands above use the study seed, and the
Zenodo source-data archive contains the values reported in the manuscript,
including retained fold-level results where applicable.

## Scope

The public entry points directly support iSCALE training and evaluation. The
source-data archive also contains the comparison values reported in the
manuscript, but this repository does not provide a unified implementation of
every comparison baseline.
