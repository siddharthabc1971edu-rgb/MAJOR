"""Project-root-anchored paths used by the CLI, library, and dashboard."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

__all__ = ["PROJECT_ROOT", "RAW_DIR", "ARTIFACTS_DIR"]
