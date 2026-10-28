"""Same four roles built with CrewAI + a real LLM. Needs: pip install -r requirements-crew.txt and an API key."""
import os
import re
import time

from . import kb
from .demo_pipeline import RunResult
from .logger import new_run_id, record


def run_crew(topic: str) -> RunResult:
    try:
        from crewai import Agent, Crew, LLM, Process, Task
        from crewai.tools import tool
    except ImportError as e:
        raise SystemExit("CrewAI is not installed. Run: pip install -r requirements-crew.txt (or use --mode demo)") from e

    llm = LLM(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.2)
    res = RunResult(new_run_id(), topic)

    @tool("Knowledge base search")
    def kb_search(query: str) -> str:
        """Search the local knowledge base. Returns sentences tagged with their source id, e.g. [kb:agentic_ai]."""
        docs = kb.search(query, k=3)
        return "\n".join(f"[kb:{d.id}] {s}" for d in docs for s in d.sentences) or "NO_RESULTS"

    researcher = Agent(role="Researcher", goal="Collect relevant facts about {topic} with source ids",
                       backstory="Careful analyst who only reports what the knowledge base says.",
                       tools=[kb_search], llm=llm, verbose=False)
    checker = Agent(role="Fact Checker", goal="Verify every claim against its cited source and drop unsupported ones",
                    backstory="Sceptical editor. Never lets an uncited or unsupported claim through.",
                    tools=[kb_search], llm=llm, verbose=False)
    writer = Agent(role="Writer", goal="Write a short, clear research brief with numbered citations",
                   backstory="Technical writer who explains things simply.", llm=llm, verbose=False)
    reviewer = Agent(role="Reviewer", goal="Judge the brief and flag risks for the human reviewer",
                     backstory="Quality lead who is strict about citations.", llm=llm, verbose=False)

    clock = {"t": time.perf_counter()}

    def cb(role, task_name):
        def _cb(output):
            now = time.perf_counter()
            text = getattr(output, "raw", str(output))
            q = 0.0
            if role == "reviewer":
                m = re.search(r"QUALITY_SCORE:\s*([01](?:\.\d+)?)", text)
                q = float(m.group(1)) if m else 0.0
                res.quality = q
            record(res.steps, res.run_id, role, task_name, now - clock["t"], text, q)
            clock["t"] = now
        return _cb

    t1 = Task(description="Search the knowledge base for facts about: {topic}. List 6-8 claims, each ending with its source tag like [kb:agentic_ai].",
              expected_output="Bulleted claims, each with a [kb:<id>] tag.", agent=researcher,
              callback=cb("researcher", "source_collection"))
    t2 = Task(description="Check each claim against the knowledge base. Keep only claims fully supported by the cited source. List removed claims separately.",
              expected_output="Verified claims list and a removed-claims list.", agent=checker, context=[t1],
              callback=cb("fact_checker", "claim_verification"))
    t3 = Task(description="Write a brief on {topic} using only verified claims, with numbered citations [1], [2] and a Sources section.",
              expected_output="Markdown brief with Key findings and Sources.", agent=writer, context=[t2],
              callback=cb("writer", "draft_report"))
    t4 = Task(description="Review the brief for unsupported statements and missing citations. End with exactly one line: QUALITY_SCORE: <number between 0 and 1>.",
              expected_output="Short review ending with a QUALITY_SCORE line.", agent=reviewer, context=[t3],
              callback=cb("reviewer", "final_review"))

    Crew(agents=[researcher, checker, writer, reviewer], tasks=[t1, t2, t3, t4],
         process=Process.sequential, verbose=False).kickoff(inputs={"topic": topic})

    brief = t3.output.raw if t3.output else ""
    review = t4.output.raw if t4.output else ""
    res.report = (f"# Research Brief: {topic}\n_Run {res.run_id} | Status: PENDING HUMAN REVIEW_\n\n"
                  f"{brief}\n\n## Reviewer notes\n{review}\n")
    return res
