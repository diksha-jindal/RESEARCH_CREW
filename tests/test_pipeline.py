import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import kb, logger
from src.demo_pipeline import run_demo


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "LOG_PATH", tmp_path / "run_log.csv")


def test_search_finds_docs():
    assert kb.search("agentic ai")


def test_clean_run_has_citations_and_no_errors(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    r = run_demo("Agentic AI")
    assert all(c.verified for c in r.claims)
    assert "[1]" in r.report and r.quality >= 0.7
    assert len(r.steps) == 4


def test_fact_checker_catches_fake_claim(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    r = run_demo("Agentic AI", inject_fault=True)
    assert sum(not c.verified for c in r.claims) == 1
    assert "never fail" not in r.report
    assert any(s["errors"] == "hallucinated_citation" for s in r.steps)


def test_unknown_topic_is_handled(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert run_demo("zzzz qqqq").status == "NO SOURCES"
