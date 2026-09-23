"""Fit + cross-validate the KB prior and escalation model; writes models/*.json.

Usage:  python scripts/train_models.py [--folds 5] [--seeds 3]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from servewell_agent import config, training  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    w = training.load_world()
    print(f"labelled tickets: {len(w.ids)}   L2 share: {w.y.mean():.2f}")
    cv = training.cross_validate(w, a.folds, a.seeds)
    print(f"CV ({a.folds}-fold x {a.seeds} seeds): accuracy={cv['accuracy']:.3f}  L2-recall={cv['l2_recall']:.3f}  "
          f"L2-precision={cv['l2_precision']:.3f}  AUC={cv['auc']:.3f}")
    print(f"baseline rule (L1 history or flag => L2): accuracy={cv['baseline_accuracy']:.3f}  recall={cv['baseline_l2_recall']:.3f}")
    model, prior = training.fit(w.ids, w)
    model["cv"] = {k: float(v) for k, v in cv.items()}
    config.MODELS_DIR.mkdir(exist_ok=True)
    (config.MODELS_DIR / "escalation_model.json").write_text(json.dumps(model, indent=2))
    (config.MODELS_DIR / "kb_prior.json").write_text(json.dumps({"subcat_doc_prior": prior}, indent=1))
    print("\nCoefficients (standardised features):")
    for f, v in sorted(model["coef"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:14s} {v:+.2f}")
    print("\nwrote models/escalation_model.json and models/kb_prior.json")


if __name__ == "__main__":
    main()
