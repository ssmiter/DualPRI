# Reproducing the manuscript analyses

## Version rule

Use a versioned code release, a versioned data record, and a checkpoint produced
from the same configuration. Do not mix files from the moving `master` branch
with a published data DOI.

The release candidate configuration is stored in:

- `config.PAPER_MODEL_CONFIG` for Python entry points;
- `configs/paper_config.yaml` for the human-readable record.

Before publishing `v1.0.0`, compare both files with the saved configuration of
the final checkpoint and fill in the checkpoint checksum and persistent IDs.

## Evaluation groups

The manuscript evaluates four related groups of evidence:

1. protein–RNA binding-affinity prediction under standard, reduced-data, and
   structure-aware splitting settings;
2. sensitivity to multiscale distance definitions and model hyperparameters;
3. transfer to protein stability and protein–protein binding benchmarks;
4. interpretation case studies combining learned interaction scores with
   structural and molecular-dynamics analyses.

The figure source-data record contains the values used to prepare the final
plots. It is not a replacement for the model-ready training data or checkpoint.

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

Exact fold assignments or the random seeds used to generate them must be
deposited with the data record whenever a reported value depends on a fixed
split.

## Reporting checklist

For every reported experiment, retain:

- the code tag or commit;
- the data-record version;
- the complete configuration;
- fold assignments and random seeds;
- the best checkpoint and its SHA-256 checksum;
- per-fold or per-seed predictions and metrics;
- the aggregation rule used for the displayed mean and standard deviation.

## Known release boundary

Implementations of all comparison baselines are not part of the current public
entry point. The repository release should claim direct reproduction of iSCALE,
while the source-data record preserves the reported comparison values and their
provenance. Add baseline code only if it is complete, licensed for redistribution,
and tied to the final experiment configuration.
