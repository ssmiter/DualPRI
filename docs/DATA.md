# Data access and provenance

## Release boundary

The GitHub repository contains code, configuration, documentation, and small
benchmark metadata tables. Large processed datasets, checkpoints, molecular-
dynamics trajectories, and figure source data are released separately so that
the software repository remains lightweight and versionable.

The public data record will be linked here before the `v1.0.0` release:

- Dataset DOI: **to be assigned**
- Dataset landing page: **to be assigned**
- Software DOI: **to be assigned through the Zenodo–GitHub integration**

## Recommended data-record files

The manuscript data record should contain:

1. `iSCALE_source_data_v1.0.zip` — source values for Figures 2–6,
   Supplementary Tables 1–3, provenance scripts, and a public README;
2. `iSCALE_pipeline_archive_v1.0.zip` or equivalent TAR.GZ archives — the
   reusable preprocessing layer, where third-party redistribution terms permit;
3. the final checkpoint and a minimal model-ready example;
4. `SHA256SUMS.txt` — checksums for every deposited archive.

Full molecular-dynamics trajectories and simulation inputs may be deposited as
a separate related record because they are substantially larger than the figure
source package.

## Expected processed dataset

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

The preprocessing utilities under `dataset_process/` operate on mutation
metadata, experimentally resolved complex structures, sequence profiles, and
residue-level conservation values. Users are responsible for obtaining
third-party resources under their original terms and for citing the underlying
databases and benchmark publications.

The code release does not include machine-specific caches, private server
paths, or unrelated exploratory experiment outputs.

## Source-data relationship

The figure source package and this software release are distinct research
objects and should receive separate persistent identifiers. The data record
should link to the software DOI, and the software README should link to the
version-specific data DOI.

## Reviewer access

If a large file cannot be made public at initial submission, it should remain
available to editors and reviewers through a stable private or restricted
repository link. The final Data Availability statement must describe the access
route that actually exists at the time of submission.
