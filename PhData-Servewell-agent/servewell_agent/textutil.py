"""Small text utilities shared by retrieval and agents (no heavy NLP deps on purpose)."""
from __future__ import annotations

import re

STOP = set(
    "a an the and or of to in on for with is are was were be been it its this that these those at by "
    "from as if then else so do does did not no can could should would will you your we our they their "
    "i my me there here have has had but about into out up down over than too very just also any all "
    "each per via etc which when what where who how make made using use used".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-\._][a-z0-9]+)*")
_NOT_CODES = ("INC", "SW", "POS", "PRN", "RTR", "SSM", "KSK", "NL", "SC", "SG", "TM", "PA")
ERROR_CODE = re.compile(
    r"\b(?:0x[0-9A-Fa-f]{3,8}"                         # 0x8004, 0x80070005
    r"|[A-Z]{1,5}-[A-Z]{2,5}-\d{2,3}"                  # E-PRINT-502, MIX-LOW-002, CLN-RNS-03
    r"|[A-Z]{1,5}-\d{3}-\d{3}"                          # PTW-043-001
    r"|[A-Z]{1,5}-\d{3,4}"                              # E-402, ERR-4521, LSY-4002
    r"|[EF]\d{2})\b"                                    # E04, F06
)


def stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "ies":
                return tok[: -len(suf)] + "y"
            return tok[: -len(suf)]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> list:
    out = []
    for m in _TOKEN.finditer(text.lower()):
        tok = m.group(0)
        parts = re.split(r"[-\._]", tok)
        cand = [tok] + parts if len(parts) > 1 else [tok]
        for c in cand:
            if not c or (not keep_stop and c in STOP) or len(c) < 2:
                continue
            out.append(stem(c))
    return out


def content_tokens(text: str) -> set:
    return set(tokenize(text))


def find_error_codes(text: str) -> list:
    codes = []
    for m in ERROR_CODE.finditer(text):
        c = m.group(0)
        if c.split("-")[0] in _NOT_CODES:
            continue
        if c not in codes:
            codes.append(c)
    return codes
