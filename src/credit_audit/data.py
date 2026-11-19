from __future__ import annotations

import hashlib
import io
import tempfile
import warnings
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .paths import RAW_DIR


GERMAN_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
AUSTRALIAN_URL = "https://archive.ics.uci.edu/static/public/143/statlog+australian+credit+approval.zip"

# Checksums of the UCI archives cached in this repository's documented setup.
# A mismatch on a structurally valid archive is reported as a warning so that a
# future upstream revision does not silently change reported results.
EXPECTED_ZIP_SHA256 = {
    "german-credit.zip": "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b",
    "australian-credit.zip": "f0bd9cd9b1d133cb20456b8953e1f8139fd30ed2b3f66da540db75136f1e9e0b",
}

GERMAN_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker",
]
GERMAN_NUMERIC = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents",
]
GERMAN_PERTURBATION_FEATURES = list(GERMAN_NUMERIC)
AUSTRALIAN_COLUMNS = [f"A{i}" for i in range(1, 15)]
AUSTRALIAN_NUMERIC = ["A2", "A3", "A8", "A11", "A14"]
AUSTRALIAN_PERTURBATION_FEATURES = list(AUSTRALIAN_NUMERIC)

NUMERIC_BY_DATASET = {
    "german": GERMAN_NUMERIC,
    "australian": AUSTRALIAN_NUMERIC,
}
PERTURBATION_BY_DATASET = {
    "german": GERMAN_PERTURBATION_FEATURES,
    "australian": AUSTRALIAN_PERTURBATION_FEATURES,
}


def _validate_archive(content: bytes, expected_member: str, label: str) -> None:
    if not zipfile.is_zipfile(io.BytesIO(content)):
        raise ValueError(
            f"{label} is not a valid ZIP archive. Delete it and rerun to download a fresh copy."
        )
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        if not any(name.lower().endswith(expected_member.lower()) for name in zf.namelist()):
            raise ValueError(
                f"{label} does not contain {expected_member!r}. "
                "Delete it and rerun to download a fresh copy."
            )


def _check_sha256(content: bytes, expected_name: str) -> None:
    expected = EXPECTED_ZIP_SHA256.get(expected_name)
    if not expected:
        return
    actual = hashlib.sha256(content).hexdigest()
    if actual.lower() != expected.lower():
        warnings.warn(
            f"SHA-256 mismatch for cached {expected_name}: expected {expected}, got {actual}. "
            "The archive is structurally valid, but verify the source before relying on results.",
            RuntimeWarning,
            stacklevel=3,
        )


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    try:
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _download_zip(
    url: str, expected_name: str, expected_member: str, raw_dir: Path = RAW_DIR
) -> bytes:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / expected_name
    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        with urlopen(url, timeout=30) as response:
            content = response.read()

    _validate_archive(content, expected_member, str(cache_path))
    _check_sha256(content, expected_name)
    if not cache_path.exists():
        _write_atomically(cache_path, content)
    return content


def load_dataset(name: str, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load a UCI benchmark; target 1 consistently denotes the paper's bad/minority class."""
    if name == "german":
        archive = _download_zip(GERMAN_URL, "german-credit.zip", "german.data", raw_dir)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith("german.data")), None)
            if member is None:  # pragma: no cover - guarded by _validate_archive
                raise FileNotFoundError("The UCI German Credit archive did not contain german.data.")
            frame = pd.read_csv(
                io.BytesIO(zf.read(member)), sep=r"\s+", header=None,
                names=GERMAN_COLUMNS + ["raw_target"],
            )
        y = (frame.pop("raw_target") == 2).astype("int8").rename("default")
        groups = frame["personal_status_sex"].map(_german_sex).rename("sex")
        for column in GERMAN_NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame, y, groups

    if name == "australian":
        archive = _download_zip(AUSTRALIAN_URL, "australian-credit.zip", "australian.dat", raw_dir)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith("australian.dat")), None)
            if member is None:  # pragma: no cover - guarded by _validate_archive
                raise FileNotFoundError("The UCI Australian Credit archive did not contain australian.dat.")
            frame = pd.read_csv(
                io.BytesIO(zf.read(member)), sep=r"\s+", header=None,
                names=AUSTRALIAN_COLUMNS + ["raw_target"],
            )
        # The paper treats the 307/690 (44.5%) positive class as default.
        y = (pd.to_numeric(frame.pop("raw_target"), errors="raise") == 1).astype("int8").rename("default")
        groups = pd.Series("unknown", index=frame.index, name="sex")
        for column in AUSTRALIAN_NUMERIC:
            frame[column] = pd.to_numeric(frame[column].replace("?", np.nan), errors="coerce")
        frame = frame.replace("?", np.nan)
        return frame, y, groups

    raise ValueError(f"Unknown dataset {name!r}; choose 'german' or 'australian'.")


def _german_sex(value: str) -> str:
    if value in {"A92", "A95"}:
        return "female"
    if value in {"A91", "A93", "A94"}:
        return "male"
    return "unknown"
