"""Offline 4-agent pipeline (no API key needed).

Researcher -> Fact-Checker -> Writer -> Reviewer -> [human approval gate]
The agents are rule-based here so the demo is deterministic and free; crew_pipeline.py runs the
same roles with real LLMs through CrewAI.
"""
import time
from dataclasses import dataclass, field

from . import kb
from . import logger
from .logger import new_run_id, record

FAULT_CLAIM = "A 2023 benchmark proved that agentic systems never fail and need no human oversight."
SUPPORT_THRESHOLD = 0.85


@dataclass
class Claim:
    text: str
    doc_id: str
    doc_title: str
    support: float = 0.0
    verified: bool = False


@dataclass
class RunResult:
    run_id: str
    topic: str
    claims: list = field(default_factory=list)
    report: str = ""
    quality: float = 0.0
    status: str = "PENDING HUMAN REVIEW"
    steps: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    path: str = ""


def _rank_sentences(doc, query, n=3):
    q = set(kb.tokenize(query))
    ranked = sorted(enumerate(doc.sentences), key=lambda x: (-len(q & set(kb.tokenize(x[1]))), x[0]))
    return [s for _, s in ranked[:n]]


def run_demo(topic: str, inject_fault: bool = False) -> RunResult:
    res = RunResult(new_run_id(), topic)

    # 1) Researcher: retrieve sources and extract candidate claims
    t0 = time.perf_counter()
    docs = kb.search(topic)
    if not docs:
        res.status, res.report = "NO SOURCES", f"No sources found in the knowledge base for '{topic}'."
        res.flags.append("no_sources")
        record(res.steps, res.run_id, "researcher", "source_collection", time.perf_counter() - t0,
               res.report, 0.0, "tool_failure")
        return res
    for d in docs:
        for s in _rank_sentences(d, topic):
            res.claims.append(Claim(s, d.id, d.title))
    if inject_fault:
        res.claims.append(Claim(FAULT_CLAIM, docs[0].id, docs[0].title))
    record(res.steps, res.run_id, "researcher", "source_collection", time.perf_counter() - t0,
           " ".join(c.text for c in res.claims), min(1.0, len(docs) / 3))

    # 2) Fact-Checker: every claim must be supported by the document it cites
    t0 = time.perf_counter()
    for c in res.claims:
        c.support = round(kb.support(c.text, kb.get_doc(c.doc_id).text), 2)
        c.verified = c.support >= SUPPORT_THRESHOLD
    ok = sum(c.verified for c in res.claims)
    bad = len(res.claims) - ok
    if bad:
        res.flags.append(f"{bad} unsupported claim(s) removed")
    record(res.steps, res.run_id, "fact_checker", "claim_verification", time.perf_counter() - t0,
           " ".join(c.text for c in res.claims), ok / len(res.claims),
           "hallucinated_citation" if bad else "none")

    # 3) Writer: compose the brief from verified claims only, with numbered citations
    t0 = time.perf_counter()
    order, lines = [], []
    for c in (c for c in res.claims if c.verified):
        if c.doc_id not in order:
            order.append(c.doc_id)
        lines.append(f"{len(lines) + 1}. {c.text} [{order.index(c.doc_id) + 1}]")
    titles = {d.id: d.title for d in docs}
    sources = [f"[{i + 1}] {titles.get(d, d)} (kb:{d})" for i, d in enumerate(order)]
    res.report = (
        f"# Research Brief: {topic}\n"
        f"_Run {res.run_id} | Status: PENDING HUMAN REVIEW_\n\n"
        "## Key findings\n" + "\n".join(lines) + "\n\n"
        "## Sources\n" + "\n".join(sources) + "\n\n"
        "## Fact-check summary\n"
        f"- Verified claims: {ok} of {len(res.claims)}\n"
        f"- Excluded as unsupported: {bad}\n"
    )
    record(res.steps, res.run_id, "writer", "draft_report", time.perf_counter() - t0,
           res.report, 1.0 if ok else 0.0)

    # 4) Reviewer: heuristic quality score and flags for the human reviewer
    t0 = time.perf_counter()
    res.quality = round(0.5 * ok / len(res.claims) + 0.3 * min(1.0, ok / 6) + 0.2 * (1.0 if len(order) >= 2 else 0.5), 2)
    if res.quality < 0.7:
        res.flags.append("low quality score")
    record(res.steps, res.run_id, "reviewer", "final_review", time.perf_counter() - t0,
           res.report, res.quality)
    return res


def finalize(res: RunResult, approved: bool) -> str:
    """Human-in-the-loop gate: only approved briefs go to outputs/reports, others stay drafts."""
    res.status = "APPROVED" if approved else "REJECTED (DRAFT)"
    folder = logger.ROOT / "outputs" / ("reports" if approved else "drafts")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{res.run_id}.md"
    path.write_text(res.report.replace("PENDING HUMAN REVIEW", res.status), encoding="utf-8")
    res.path = str(path)
    return res.path
