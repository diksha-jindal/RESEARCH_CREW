"""Run logging. Same columns as data/dataset_additional.csv so live runs can join the analytics."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "outputs" / "run_log.csv"
COLUMNS = ["run_id", "agent_role", "task_type", "tokens_used", "duration_s", "quality_score", "errors"]


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def new_run_id() -> str:
    n = 1
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            n = len({r["run_id"] for r in csv.DictReader(f)}) + 1
    return f"LIVE{n:05d}"


def record(steps: list, run_id, role, task, duration, text, quality, error="none") -> None:
    row = {
        "run_id": run_id, "agent_role": role, "task_type": task,
        "tokens_used": estimate_tokens(text), "duration_s": round(duration, 3),
        "quality_score": round(quality, 2), "errors": error,
    }
    steps.append(row)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)
