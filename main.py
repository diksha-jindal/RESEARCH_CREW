"""CLI:  python main.py "Agentic AI" [--mode demo|crew] [--inject-fault] [--approve ask|yes|no]"""
import argparse

from dotenv import load_dotenv

from src.demo_pipeline import finalize, run_demo

load_dotenv()


def main():
    ap = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    ap.add_argument("topic", nargs="?", default="Agentic AI")
    ap.add_argument("--mode", choices=["demo", "crew"], default="demo", help="demo = offline, crew = CrewAI + LLM")
    ap.add_argument("--inject-fault", action="store_true", help="add a fake claim to show the fact-checker catching it")
    ap.add_argument("--approve", choices=["ask", "yes", "no"], default="ask", help="human review decision")
    a = ap.parse_args()

    if a.mode == "crew":
        from src.crew_pipeline import run_crew
        res = run_crew(a.topic)
    else:
        res = run_demo(a.topic, a.inject_fault)

    print(f"\nRun {res.run_id} | topic: {a.topic}\n")
    print(f"{'agent':<14}{'task':<20}{'tokens':>8}{'secs':>9}{'quality':>9}  errors")
    for s in res.steps:
        print(f"{s['agent_role']:<14}{s['task_type']:<20}{s['tokens_used']:>8}{s['duration_s']:>9}{s['quality_score']:>9}  {s['errors']}")
    for c in res.claims:
        print(f"  [{'OK ' if c.verified else 'BAD'}] support={c.support:<5} {c.text[:90]}")
    print("\n" + res.report)
    if res.flags:
        print("Flags:", "; ".join(res.flags))
    print(f"Quality score: {res.quality}")

    if res.status == "NO SOURCES":
        return
    if a.approve == "ask":
        try:
            ok = input("Human review - approve for publishing? [y/N]: ").strip().lower() == "y"
        except EOFError:
            ok = False
    else:
        ok = a.approve == "yes"
    print(f"{'Approved' if ok else 'Rejected'} -> saved to {finalize(res, ok)}")


if __name__ == "__main__":
    main()
