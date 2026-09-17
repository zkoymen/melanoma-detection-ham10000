from pathlib import Path

import numpy as np
import pandas as pd

from src.clean_dataset import audit_raw_overlap, build_clean_isic_master_dataset


def test_clean_builder_uses_metadata_and_fallback_groups(tmp_path: Path):
    legacy = tmp_path / "legacy"
    clean = tmp_path / "clean"
    legacy.mkdir()
    ham = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    np.save(legacy / "X_all.npy", ham)
    np.save(legacy / "y_all.npy", np.array([1, 1, 0, 0]))
    np.save(legacy / "ids_all.npy", np.array(["H1", "H2", "H3", "H4"], dtype=object))
    np.save(legacy / "lesion_ids_all.npy", np.array(["LH1", "LH2", "LH3", "LH4"], dtype=object))

    np.save(legacy / "X_isic2019_mel.npy", np.stack([ham[0], np.full_like(ham[0], 9), np.full_like(ham[0], 10)]))
    np.save(legacy / "ids_isic2019_mel.npy", np.array(["H1", "I2", "I3"], dtype=object))
    np.save(legacy / "X_isic2019_nonmel.npy", np.stack([ham[2], np.full_like(ham[2], 11), np.full_like(ham[2], 12)]))
    np.save(legacy / "ids_isic2019_nonmel.npy", np.array(["H3", "I4", "I5"], dtype=object))
    pd.DataFrame({
        "image": ["H1", "I2", "I3", "H3", "I4", "I5"],
        "lesion_id": ["L1", np.nan, "L3", "L4", "L5", "L6"],
    }).to_csv(tmp_path / "metadata.csv", index=False)

    audit = audit_raw_overlap(legacy, legacy, tmp_path / "metadata.csv")
    assert audit["exact_image_id_overlap_count"] == 2
    assert audit["isic_ids_with_missing_lesion_id_count"] == 1

    _, _, ids, train, val, test, manifest = build_clean_isic_master_dataset(
        legacy, legacy, tmp_path / "metadata.csv", clean, seed=7
    )
    assert len(ids) == 6
    assert manifest["checks"]["missing_metadata_rows"] == 0
    assert manifest["checks"]["missing_lesion_ids_using_image_fallback"] == 1
    assert set(ids[train]).isdisjoint(set(ids[val]))
    assert set(ids[train]).isdisjoint(set(ids[test]))
    assert set(ids[val]).isdisjoint(set(ids[test]))
