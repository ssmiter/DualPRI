# Data access

## Public data record

The processed protein-RNA dataset and the source data supporting the manuscript
are archived on Zenodo:

- DOI: [10.5281/zenodo.21972360](https://doi.org/10.5281/zenodo.21972360)
- `protein_rna_dataset.pkl`: the processed dataset used by the training and
  evaluation entry points;
- `iSCALE_source_data.zip`: source values and retained fold-level results for
  Figures 2-6 and Supplementary Tables 1-3.

The source-data archive supports the reported figures and tables. It is not an
archive of every intermediate file produced during model development.

## Processed dataset

Training uses a pickle file containing a list of mutation samples. Each sample
must provide compatible wild-type and mutant protein graphs, binding-partner
information, mutation metadata, and an experimental affinity-change target.
The default location is:

```text
data/processed/protein_rna_dataset.pkl
```

An alternative path can be passed with `--data_path` or set with the
`ISCALE_DATA_PATH` environment variable.

## Preprocessing inputs

The utilities under `dataset_process/` operate on mutation metadata,
experimentally resolved complex structures, sequence profiles, and
residue-level conservation values. Users are responsible for obtaining
third-party resources under their original terms and for citing the underlying
databases and benchmark publications.

The repository does not include machine-specific caches, private server paths,
or unrelated exploratory outputs.
