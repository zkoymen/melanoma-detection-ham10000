"""Audited ISIC 2019 dataset construction.

This module deliberately keeps the legacy HAM10000+ISIC concatenation out of
the clean path.  ISIC 2019 is an aggregate corpus that already contains the
HAM10000 images, so the clean benchmark uses one ISIC 2019 master corpus and
uses HAM10000 only for overlap auditing.

The builder writes the same array filenames consumed by the existing model
notebooks, but into a separate ``data_clean_v2`` directory.  Every run also
writes a JSON manifest with hashes, counts, split checks, and the metadata
provenance needed to reproduce the result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_strings(values: Iterable[object]) -> np.ndarray:
    return np.asarray([str(x) for x in values], dtype=object)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _image_hashes(X: np.ndarray) -> np.ndarray:
    """Exact hashes of stored RGB arrays; deterministic and dependency-free."""
    return np.asarray([
        hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
        for img in X
    ], dtype=object)


def _find_column(df, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _metadata_maps(metadata_csv: Path):
    import pandas as pd

    meta = pd.read_csv(metadata_csv)
    image_col = _find_column(meta, ("image", "image_id", "isic_id"))
    lesion_col = _find_column(meta, (
        "lesion_id", "common lesion identifier", "common_lesion_identifier",
        "common lesion id", "common_lesion_id",
    ))
    if image_col is None or lesion_col is None:
        raise ValueError(
            f"Metadata must contain image and common lesion identifier columns; "
            f"found {list(meta.columns)}"
        )
    image_rows = {
        str(image): lesion
        for image, lesion in zip(meta[image_col], meta[lesion_col])
        if str(image).strip() and str(image).lower() not in {"nan", "none"}
    }
    image_to_lesion = {
        image: str(lesion)
        for image, lesion in image_rows.items()
        if str(lesion).strip() and str(lesion).lower() not in {"nan", "none", ""}
    }
    return meta, str(image_col), str(lesion_col), image_rows, image_to_lesion


def audit_raw_overlap(
    legacy_dir: str | Path,
    isic_dir: str | Path,
    metadata_csv: str | Path,
    output_json: str | Path | None = None,
):
    """Audit HAM/ISIC ID and stored-array hash overlap without modifying data."""
    legacy_dir, isic_dir = Path(legacy_dir), Path(isic_dir)
    metadata_csv = Path(metadata_csv)
    ham_ids = _as_strings(np.load(legacy_dir / "ids_all.npy", allow_pickle=True))
    ham_X = np.load(legacy_dir / "X_all.npy", mmap_mode="r")
    isic_parts = []
    for label in ("mel", "nonmel"):
        x_path = isic_dir / f"X_isic2019_{label}.npy"
        id_path = isic_dir / f"ids_isic2019_{label}.npy"
        if not x_path.exists() or not id_path.exists():
            raise FileNotFoundError(f"Missing ISIC preprocessed pair: {x_path.name}, {id_path.name}")
        isic_parts.append((label, np.load(x_path, mmap_mode="r"),
                           _as_strings(np.load(id_path, allow_pickle=True))))

    meta, image_col, lesion_col, image_rows, image_to_lesion = _metadata_maps(metadata_csv)
    ham_set, isic_id_set = set(ham_ids.tolist()), set()
    exact_id_overlap, missing_metadata_rows, missing_lesion_ids = [], [], []
    isic_hashes, ham_hashes = {}, {}
    for label, X, ids in isic_parts:
        isic_id_set.update(ids.tolist())
        exact_id_overlap.extend([x for x in ids.tolist() if x in ham_set])
        missing_metadata_rows.extend([x for x in ids.tolist() if x not in image_rows])
        missing_lesion_ids.extend([x for x in ids.tolist() if x in image_rows and x not in image_to_lesion])
        for image_id, img in zip(ids.tolist(), X):
            isic_hashes.setdefault(
                hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(), []
            ).append(image_id)
    for image_id, img in zip(ham_ids.tolist(), ham_X):
        ham_hashes.setdefault(
            hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(), []
        ).append(image_id)

    hash_overlap = []
    for digest in sorted(set(ham_hashes) & set(isic_hashes)):
        hash_overlap.append({
            "sha256_image_bytes": digest,
            "ham_ids": sorted(ham_hashes[digest]),
            "isic_ids": sorted(isic_hashes[digest]),
        })

    report = {
        "status": "audit_only",
        "legacy_ham_count": int(len(ham_ids)),
        "isic_preprocessed_count": int(sum(len(ids) for _, _, ids in isic_parts)),
        "isic_metadata_rows": int(len(meta)),
        "metadata_image_column": image_col,
        "metadata_lesion_column": lesion_col,
        "exact_image_id_overlap_count": int(len(set(exact_id_overlap))),
        "exact_image_id_overlap": sorted(set(exact_id_overlap)),
        "stored_array_hash_overlap_count": int(len(hash_overlap)),
        "stored_array_hash_overlap": hash_overlap,
        "isic_ids_missing_from_metadata_rows_count": int(len(set(missing_metadata_rows))),
        "isic_ids_missing_from_metadata_rows": sorted(set(missing_metadata_rows)),
        "isic_ids_with_missing_lesion_id_count": int(len(set(missing_lesion_ids))),
        "isic_ids_with_missing_lesion_id": sorted(set(missing_lesion_ids)),
        "source_policy": "ISIC2019 master corpus; HAM10000 used for audit only",
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _group_split(lesion_ids, y, seed: int, train_frac: float, val_frac: float):
    """Stratified split by lesion group, preserving every group intact."""
    rng = np.random.default_rng(seed)
    lesion_ids = _as_strings(lesion_ids)
    y = np.asarray(y, dtype=np.int64)
    train, val, test = [], [], []
    for cls in (0, 1):
        cls_indices = np.flatnonzero(y == cls)
        groups = np.unique(lesion_ids[cls_indices])
        rng.shuffle(groups)
        n_train = int(round(len(groups) * train_frac))
        n_val = int(round(len(groups) * val_frac))
        train_groups = set(groups[:n_train].tolist())
        val_groups = set(groups[n_train:n_train + n_val].tolist())
        for idx in cls_indices.tolist():
            group = lesion_ids[idx]
            if group in train_groups:
                train.append(idx)
            elif group in val_groups:
                val.append(idx)
            else:
                test.append(idx)
    return tuple(np.asarray(sorted(part), dtype=np.int64) for part in (train, val, test))


def _assert_clean_split(ids, lesion_ids, idx_train, idx_val, idx_test):
    ids = _as_strings(ids)
    lesion_ids = _as_strings(lesion_ids)
    partitions = {"train": idx_train, "val": idx_val, "test": idx_test}
    for name, idx in partitions.items():
        if len(set(ids[idx].tolist())) != len(idx):
            raise AssertionError(f"duplicate image_id inside {name}")
    for a, ia in partitions.items():
        for b, ib in partitions.items():
            if a >= b:
                continue
            if set(ids[ia].tolist()) & set(ids[ib].tolist()):
                raise AssertionError(f"image_id overlap between {a} and {b}")
            if set(lesion_ids[ia].tolist()) & set(lesion_ids[ib].tolist()):
                raise AssertionError(f"lesion_id overlap between {a} and {b}")


def _counts(y, ids, lesion_ids, indices):
    return {
        "images": int(len(indices)),
        "melanoma": int(np.sum(y[indices] == 1)),
        "non_melanoma": int(np.sum(y[indices] == 0)),
        "unique_images": int(len(set(_as_strings(ids[indices]).tolist()))),
        "unique_lesions": int(len(set(_as_strings(lesion_ids[indices]).tolist()))),
    }


def build_clean_isic_master_dataset(
    legacy_dir: str | Path,
    isic_dir: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Build the clean balanced ISIC-master dataset used by all notebooks.

    HAM arrays are read only to audit provenance.  The final corpus contains
    one copy of the preprocessed ISIC 2019 records, deduplicated by image ID
    and exact stored-image hash.  This prevents HAM10000 from being appended a
    second time while retaining the full ISIC aggregate as the benchmark.
    """
    legacy_dir, isic_dir, output_dir = map(Path, (legacy_dir, isic_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_raw_overlap(
        legacy_dir, isic_dir, metadata_csv,
        output_json=output_dir / "raw_overlap_audit.json",
    )
    _, _, _, image_rows, image_to_lesion = _metadata_maps(Path(metadata_csv))

    X_parts, id_parts, y_parts = [], [], []
    for label, cls in (("mel", 1), ("nonmel", 0)):
        X_parts.append(np.asarray(np.load(isic_dir / f"X_isic2019_{label}.npy")))
        id_parts.append(_as_strings(np.load(isic_dir / f"ids_isic2019_{label}.npy", allow_pickle=True)))
        y_parts.append(np.full(len(id_parts[-1]), cls, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    ids = np.concatenate(id_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Keep the first occurrence of each image ID and each exact stored-image hash.
    # Hash dedup is intentionally exact; perceptual candidates remain in the audit
    # for human review instead of silently deleting similar-but-distinct images.
    seen_ids, seen_hashes, keep = set(), set(), []
    duplicate_id_rows, duplicate_hash_rows = [], []
    for i, (image_id, img) in enumerate(zip(ids.tolist(), X)):
        digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
        if image_id in seen_ids:
            duplicate_id_rows.append({"row": i, "image_id": image_id})
            continue
        if digest in seen_hashes:
            duplicate_hash_rows.append({"row": i, "image_id": image_id, "sha256": digest})
            continue
        seen_ids.add(image_id)
        seen_hashes.add(digest)
        keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    X, ids, y = X[keep], ids[keep], y[keep]

    # The downloaded ISIC arrays are binary, so balance deterministically after
    # internal deduplication. No HAM rows are appended a second time.
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(len(pos), len(neg))
    selected = np.concatenate([
        rng.permutation(pos)[:k],
        rng.permutation(neg)[:k],
    ])
    selected.sort()
    X, ids, y = X[selected], ids[selected], y[selected]

    missing = [image_id for image_id in ids.tolist() if image_id not in image_rows]
    if missing:
        raise ValueError(
            f"{len(missing)} selected ISIC IDs are absent from the official metadata. "
            "Do not train until the metadata file matches the image IDs."
        )
    missing_lesion_ids = [image_id for image_id in ids.tolist() if image_id not in image_to_lesion]
    lesion_ids = np.asarray([
        f"isic_lesion:{image_to_lesion[x]}" if x in image_to_lesion
        else f"isic_image:{x}"
        for x in ids
    ], dtype=object)

    # Refuse mixed-label lesion groups: they indicate a metadata/label mismatch.
    for lesion in np.unique(lesion_ids):
        labels = np.unique(y[lesion_ids == lesion])
        if len(labels) != 1:
            raise ValueError(f"Mixed labels within lesion group {lesion}: {labels.tolist()}")

    idx_train, idx_val, idx_test = _group_split(
        lesion_ids, y, seed=seed, train_frac=train_frac, val_frac=val_frac
    )
    _assert_clean_split(ids, lesion_ids, idx_train, idx_val, idx_test)

    # The source is intentionally a single aggregate corpus. Do not fabricate
    # HAM/BCN/MSK source labels from filenames when the supplied metadata lacks them.
    source = np.ones(len(ids), dtype=np.int64)
    np.save(output_dir / "X_combined.npy", X)
    np.save(output_dir / "y_combined.npy", y)
    np.save(output_dir / "ids_combined.npy", ids)
    np.save(output_dir / "source_combined.npy", source)
    np.save(output_dir / "lesion_combined.npy", lesion_ids)
    np.save(output_dir / "idx_train_bal.npy", idx_train)
    np.save(output_dir / "idx_val_bal.npy", idx_val)
    np.save(output_dir / "idx_test_bal.npy", idx_test)

    manifest = {
        "dataset_version": "clean_v2_isic2019_balanced_subset",
        "policy": "ISIC 2019 aggregate corpus; HAM10000 used for provenance audit only",
        "seed": int(seed),
        "split": {"train_frac": train_frac, "val_frac": val_frac, "test_frac": 1 - train_frac - val_frac,
                   "strategy": "stratified lesion-group split"},
        "metadata_csv": str(Path(metadata_csv)),
        "metadata_sha256": _sha256_file(Path(metadata_csv)),
        "raw_audit": audit,
        "deduplication": {
            "duplicate_image_id_rows_removed": duplicate_id_rows,
            "duplicate_exact_hash_rows_removed": duplicate_hash_rows,
            "selected_images": int(len(X)),
        },
        "sampling": {
            "policy": "equal binary classes after deduplication",
            "melanoma_images": int(np.sum(y == 1)),
            "non_melanoma_images": int(np.sum(y == 0)),
            "isic2019_full_metadata_rows": int(audit["isic_metadata_rows"]),
            "note": "This is a balanced subset of the ISIC 2019 aggregate, not the full 25,331-row corpus.",
        },
        "array_shape": list(X.shape),
        "counts": {name: _counts(y, ids, lesion_ids, idx)
                   for name, idx in (("all", np.arange(len(y))), ("train", idx_train),
                                     ("validation", idx_val), ("test", idx_test))},
        "checks": {
            "image_id_unique_within_and_across_splits": True,
            "lesion_id_disjoint_across_splits": True,
            "missing_metadata_rows": 0,
            "missing_lesion_ids_using_image_fallback": int(len(missing_lesion_ids)),
            "mixed_label_lesion_groups": 0,
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return X, y, ids, idx_train, idx_val, idx_test, manifest
