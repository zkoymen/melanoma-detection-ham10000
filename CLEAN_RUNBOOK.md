# Clean rerun runbook

Use branch `codex/clean-data-protocol`.

1. Open `notebooks/12_isic2019_prep.ipynb` and run cells 1–7 only. This creates
   the preprocessed ISIC arrays in the legacy `data/` folder. Do not run its
   final concatenation cell.
2. Open `notebooks/13_clean_dataset_v2.ipynb` and run every cell in order.
   It downloads the official ISIC 2019 metadata, audits HAM/ISIC overlap, and
   writes the clean arrays to `data_clean_v2/`.
3. Stop after the integrity cell. Inspect `data_clean_v2/raw_overlap_audit.json`
   and `data_clean_v2/dataset_manifest.json`. Training must not start if the
   metadata row count is not 25,331 or if any selected image ID is absent.
4. Run notebooks 01–11 in their numbered order. Their preambles now select
   `data_clean_v2`, `results_clean_v2`, `checkpoints_clean_v2`, and
   `paper_clean_v2` automatically.

The legacy `data/`, `results/`, `checkpoints/`, and `paper/` directories are
not deleted or overwritten. Every metric JSON produced by the corrected run
contains `dataset_version = clean_v2_isic2019_balanced_subset`.
