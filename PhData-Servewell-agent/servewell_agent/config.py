"""Central configuration: dataset discovery, thresholds, feature flags.

Everything that a reviewer might ask "why this number?" about lives here so it can be
defended (and tuned) in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "out"


def find_data_root() -> Path:
    """Locate the synthetic dataset (folder that contains kb/, data/, tickets/).

    Order: $SERVEWELL_DATA, then a few conventional locations relative to the project.
    """
    env = os.environ.get("SERVEWELL_DATA")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJECT_ROOT / "PhData" / "PhData",
        PROJECT_ROOT / "PhData",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT.parent / "PhData" / "PhData",
        PROJECT_ROOT.parent / "PhData",
        Path.cwd() / "PhData",
    ]
    for c in candidates:
        if (c / "kb").is_dir() and (c / "data").is_dir():
            return c
    raise FileNotFoundError(
        "Could not find the dataset (needs kb/ and data/). Set SERVEWELL_DATA=/path/to/PhData/PhData"
    )


# ---- Business rules (from the SOP / SLA matrix in the dataset) -------------------------
PEAK_WINDOWS = [(11, 14), (18, 21)]          # SOP-IT-L1-002 "Priority Override Rules"
RECURRENCE_WINDOW_DAYS = 30
RECURRENCE_THRESHOLD = 2                      # "more than twice in 30 days"
PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]

# ---- Retrieval ----------------------------------------------------------------------
TOP_DOCS = 5
MIN_PRIMARY_RUNBOOK_SCORE = 0.30              # below this we refuse to ground on a runbook
SPEC_DOC_BOOST = 0.15

# ---- Guardrails -----------------------------------------------------------------------
GROUNDING_MIN_OVERLAP_EXTRACTIVE = 0.60
GROUNDING_MIN_OVERLAP_LLM = 0.35
MIN_GROUNDED_STEPS = 2
ESCALATION_THRESHOLD = 0.50                   # tuned on train via cross-validation (see eval)
LOW_CONFIDENCE = 0.55                         # below this => human review queue

# ---- LLM ------------------------------------------------------------------------------
LLM_MODEL = os.environ.get("SERVEWELL_MODEL", "claude-sonnet-5")
LLM_MAX_TOKENS = 1500
