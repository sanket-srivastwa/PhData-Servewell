"""Fit the two small learned components (used by scripts/train_models.py and eval --holdout).

  * KB prior      subcategory -> {doc: share of that subcategory's tickets that needed the doc}
  * Escalation LR interpretable logistic regression over named features (see agents/policy.py)

Everything label-derived inside a fold (prior, subcategory target encoding, scaler, classifier)
is rebuilt from that fold's *training rows only*, so cross-validated numbers are not leaky.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .agents.knowledge import Retriever
from .agents.policy import FEATURES, extract_features
from .agents.triage import TriageAgent
from .ingest import load_history_corpus, load_ticket_dir
from .intel_platform.kb import KnowledgeBase
from .intel_platform.structured import StructuredData

SOP = "sop/escalation-procedure.md"


@dataclass
class World:
    root: Path
    kb: KnowledgeBase
    sd: StructuredData
    labels: dict
    tickets: dict
    enrich: dict
    rt: Retriever
    ids: list
    y: np.ndarray


def load_world(root=None) -> World:
    root = Path(root or config.find_data_root())
    kb = KnowledgeBase(root / "kb")
    sd = StructuredData(root / "data")
    sd.register_tickets(load_history_corpus(root))
    labels = {x["ticket_id"]: x for x in json.loads((root / "labels" / "train_labels.json").read_text())}
    tickets = {t.ticket_id: t for t in load_ticket_dir(root / "tickets" / "train") if t.ticket_id in labels}
    triage = TriageAgent(sd, kb)
    enrich = {i: triage.run(t) for i, t in tickets.items()}
    ids = sorted(tickets)
    y = np.array([1 if labels[i]["correct_routing"] == "l2_escalation" else 0 for i in ids])
    return World(root, kb, sd, labels, tickets, enrich, Retriever(kb), ids, y)


def build_prior(ids, w: World) -> dict:
    per = defaultdict(list)
    for i in ids:
        per[w.tickets[i].subcategory].append(set(w.labels[i]["relevant_kb_docs"]))
    prior = {}
    for sub, sets in per.items():
        c = Counter(d for s in sets for d in s if d != SOP)
        prior[sub] = {d: round(n / len(sets), 3) for d, n in c.items()}
    return prior


def _matrix(rows, w: World, prior: dict, rate_fn) -> np.ndarray:
    w.kb.prior = prior
    X = []
    for i in rows:
        t, e = w.tickets[i], w.enrich[i]
        r = w.rt.retrieve(t, e, escalating=False)
        hits = w.rt.escalation_hits(r.primary, t, e) if r.primary else []
        f = extract_features(t, e, len(hits), rate_fn(i, t.subcategory))
        X.append([f[k] for k in FEATURES])
    return np.array(X)


def _rates(train_ids, w: World):
    yi = {i: int(v) for i, v in zip(w.ids, w.y)}
    c, n = Counter(), Counter()
    for i in train_ids:
        c[w.tickets[i].subcategory] += yi[i]
        n[w.tickets[i].subcategory] += 1
    glob = float(np.mean([yi[i] for i in train_ids]))
    return c, n, glob, yi


def fit(train_ids, w: World, C: float = 0.3) -> tuple:
    """Final fit on `train_ids`. Returns (model_dict, prior_dict)."""
    train_ids = list(train_ids)
    prior = build_prior(train_ids, w)
    c, n, glob, yi = _rates(train_ids, w)
    loo = lambda i, sub: (c[sub] - yi[i] + 2 * glob) / (n[sub] - 1 + 2)          # leave-one-out target encoding
    X = _matrix(train_ids, w, prior, loo)
    y = np.array([yi[i] for i in train_ids])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    model = {
        "features": FEATURES,
        "coef": {f: float(v) for f, v in zip(FEATURES, clf.coef_[0])},
        "mean": {f: float(m) for f, m in zip(FEATURES, mu)},
        "std": {f: float(s) for f, s in zip(FEATURES, sd)},
        "intercept": float(clf.intercept_[0]),
        "subcat_rate": {s: float((c[s] + 2 * glob) / (n[s] + 2)) for s in n},
        "global_rate": glob,
        "trained_on": len(train_ids),
    }
    return model, prior


def cross_validate(w: World, folds: int = 5, seeds: int = 3, C: float = 0.3) -> dict:
    res = []
    for seed in range(seeds):
        pred = np.zeros(len(w.ids))
        for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(w.ids, w.y):
            tr_ids = [w.ids[j] for j in tr]
            prior = build_prior(tr_ids, w)
            c, n, glob, yi = _rates(tr_ids, w)
            trs = set(tr_ids)
            rate = lambda i, sub: ((c[sub] - (yi[i] if i in trs else 0) + 2 * glob) / (n[sub] - (1 if i in trs else 0) + 2))
            Xtr, Xte = _matrix(tr_ids, w, prior, rate), _matrix([w.ids[j] for j in te], w, prior, rate)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mu) / sd, w.y[tr])
            pred[te] = clf.predict_proba((Xte - mu) / sd)[:, 1]
        hard = pred >= config.ESCALATION_THRESHOLD
        tp = (hard & (w.y == 1)).sum()
        res.append((float((hard == w.y).mean()), float(tp / w.y.sum()), float(tp / max(1, hard.sum())), float(roc_auc_score(w.y, pred))))
    a = np.array(res).mean(0)
    base = np.array([1 if (w.tickets[i].ticket_history or w.tickets[i].escalation_flag) else 0 for i in w.ids])
    return {"accuracy": a[0], "l2_recall": a[1], "l2_precision": a[2], "auc": a[3], "folds": folds, "seeds": seeds,
            "baseline_accuracy": float((base == w.y).mean()), "baseline_l2_recall": float(base[w.y == 1].mean())}
