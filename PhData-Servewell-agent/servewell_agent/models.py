"""Typed data contracts passed between agents (the "handoff" schema)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Ticket:
    ticket_id: str
    subject: str = ""
    description: str = ""
    store_id: Optional[str] = None
    store_name: Optional[str] = None
    submitted_by: Optional[str] = None
    contact_phone: Optional[str] = None
    date_opened: Optional[datetime] = None
    priority: str = "P3"
    category: str = ""
    subcategory: str = ""
    asset_id: Optional[str] = None
    system_version: Optional[str] = None
    tags: list = field(default_factory=list)
    escalation_flag: bool = False
    related_tickets: list = field(default_factory=list)
    ticket_history: list = field(default_factory=list)
    status: str = "Open"
    # NOTE: `resolution_notes` from the dataset is deliberately NOT carried here.
    # In the training data it contains the answer (label leakage). See ingest.py.
    leak_guard_dropped: list = field(default_factory=list)
    raw_missing_fields: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.subject}. {self.description}".strip()

    @property
    def history_text(self) -> str:
        return " ".join(h.get("note", "") for h in self.ticket_history if isinstance(h, dict))


@dataclass
class Finding:
    """One enrichment / validation observation, with evidence (auditable)."""
    code: str
    severity: str          # info | warn | block
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Enrichment:
    findings: list = field(default_factory=list)
    store: Optional[dict] = None
    asset: Optional[dict] = None
    resolved_asset: Optional[dict] = None       # best-guess asset if ticket asset is wrong/missing
    version_info: Optional[dict] = None
    domain: str = ""                              # effective domain after re-classification
    claimed_domain: str = ""
    error_codes: list = field(default_factory=list)
    tried_steps: list = field(default_factory=list)   # canonical steps already attempted
    dissatisfied: bool = False
    vague: bool = False
    vague_score: int = 0
    non_it: str = "none"                          # none | partial | dominant
    non_it_terms: list = field(default_factory=list)
    injection: bool = False
    recurrence_count: int = 0
    prior_tickets: list = field(default_factory=list)
    suggested_priority: str = "P3"
    priority_reasons: list = field(default_factory=list)
    sla: dict = field(default_factory=dict)
    impact: str = "single_device"                 # single_device | multiple_devices | store_wide
    missing_mandatory: list = field(default_factory=list)
    issue_core: str = ""                          # de-emotionalised statement of the technical problem
    llm_flags: list = field(default_factory=list)  # cautious signals raised by the LLM reader (force human review)

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.findings)

    def get(self, code: str):
        for f in self.findings:
            if f.code == code:
                return f
        return None


@dataclass
class Step:
    n: int
    title: str
    text: str
    source_id: str            # chunk id it is grounded on
    source_label: str
    kind: str = "guide"       # immediate | diagnosis | resolution | note
    already_tried: bool = False
    grounded: bool = True
    grounding_score: float = 1.0
    extra_sources: list = field(default_factory=list)


@dataclass
class Decision:
    route: str                                    # l1_self_service | l1_guided | l2_escalation | non_it | needs_clarification
    reasons: list = field(default_factory=list)
    escalation_score: float = 0.0
    hard_triggers: list = field(default_factory=list)
    confidence: float = 0.0
    autonomy: str = "A1_guide"
    human_review: bool = False
    review_reasons: list = field(default_factory=list)


@dataclass
class ProposedAction:
    action: str
    target: str
    params: dict
    risk: str                                     # read_only | reversible | state_changing
    requires_approval: bool
    status: str = "proposed"                      # proposed | auto_executed_simulated | approved_simulated | blocked
    rationale: str = ""
    result: Optional[str] = None


@dataclass
class AgentResult:
    ticket_id: str
    decision: Decision
    enrichment: Enrichment
    retrieved: list
    steps: list
    questions: list
    requester_message: str
    ticket_note: str
    l2_handoff: Optional[dict]
    actions: list
    guardrails: dict
    trace: list
    mode: str = "offline"
    elapsed_ms: int = 0
    llm: Optional[dict] = None                    # what the language model did on this run (reader, writer, usage)

    def to_dict(self) -> dict:
        return asdict(self)
