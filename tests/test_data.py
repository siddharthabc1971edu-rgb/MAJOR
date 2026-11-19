from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from credit_audit import data


@pytest.fixture(autouse=True)
def _skip_checksum_warning(monkeypatch):
    # Synthetic fixtures are intentionally different from the cached UCI files.
    monkeypatch.setattr(data, "EXPECTED_ZIP_SHA256", {})


def _write_zip(tmp_path, filename: str, member: str, content: str):
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, content)
    return path


def _german_text() -> str:
    rows = [
        "A11 6 A34 A41 1000 A61 A75 4 A92 A101 2 A121 30 A143 A152 1 A172 1 A191 A201 1",
        "A12 12 A32 A42 2000 A62 A73 2 A93 A101 1 A122 40 A143 A152 2 A172 2 A191 A201 2",
        "A11 8 A34 A41 1500 A61 A75 3 A95 A103 4 A121 25 A143 A152 1 A172 1 A192 A201 1",
        "A14 18 A31 A43 3000 A63 A74 1 A91 A101 2 A123 50 A143 A152 1 A172 2 A191 A201 2",
    ]
    return "\n".join(rows) + "\n"


def _australian_text() -> str:
    rows = [
        "a 30.5 10.0 u g g v 5.0 t t 1 f g 100.0 1",
        "b 22.0 5.0 ? g g v 3.0 f f 2 f g 80.0 0",
        "a 41.0 12.5 y p p v 8.0 t t ? f p 120.0 1",
        "b 35.0 8.0 y gg gg v 6.0 t f 1 t g 95.0 0",
    ]
    return "\n".join(rows) + "\n"


def test_load_german_maps_target_and_groups(tmp_path):
    _write_zip(tmp_path, "german-credit.zip", "german.data", _german_text())
    X, y, groups = data.load_dataset("german", raw_dir=tmp_path)

    assert X.shape == (4, 20)
    assert y.tolist() == [0, 1, 0, 1]
    assert groups.tolist() == ["female", "male", "female", "male"]
    assert pd.api.types.is_numeric_dtype(X["duration_months"])
    assert X["duration_months"].tolist() == [6, 12, 8, 18]
    assert data.GERMAN_PERTURBATION_FEATURES == data.GERMAN_NUMERIC


def test_load_australian_coerces_missing_values(tmp_path):
    _write_zip(tmp_path, "australian-credit.zip", "australian.dat", _australian_text())
    X, y, groups = data.load_dataset("australian", raw_dir=tmp_path)

    assert X.shape == (4, 14)
    assert y.tolist() == [1, 0, 1, 0]
    assert groups.tolist() == ["unknown"] * 4
    assert np.isnan(X.loc[1, "A4"])
    assert np.isnan(X.loc[2, "A11"])
    assert data.AUSTRALIAN_PERTURBATION_FEATURES == data.AUSTRALIAN_NUMERIC


def test_corrupt_cached_archive_reports_clear_error(tmp_path):
    (tmp_path / "german-credit.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a valid ZIP archive"):
        data.load_dataset("german", raw_dir=tmp_path)


def test_archive_without_expected_member_is_rejected(tmp_path):
    _write_zip(tmp_path, "australian-credit.zip", "wrong.dat", "a b\n")
    with pytest.raises(ValueError, match="does not contain"):
        data.load_dataset("australian", raw_dir=tmp_path)


def test_unknown_dataset_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        data.load_dataset("taiwan", raw_dir=tmp_path)
