# Multi-Agent Research Assistant (P_170 - Agentic AI)

A team of four AI agents that researches a topic, **fact-checks every claim against its cited source**,
writes a short brief with numbered citations, and stops at a **human approval gate** before anything is published.

```
Topic -> Researcher -> Fact-Checker -> Writer -> Reviewer -> Human approval -> Published brief
            |              |                                       |
      knowledge base   drops unsupported claims            approve / reject
            \______________ every step logged to run_log.csv ______________/
```

| Agent | Job |
|---|---|
| Researcher | Finds relevant sources, extracts claims with source ids |
| Fact-Checker | Checks each claim against the cited document, removes unsupported ones |
| Writer | Writes the brief using verified claims only, with [1] [2] citations |
| Reviewer | Scores quality, flags risks for the human |

## Run it (2 minutes, no API key)

```bash
pip install -r requirements.txt
python main.py "Agentic AI"                     # CLI, asks you to approve
python main.py "Agentic AI" --inject-fault      # adds a fake claim: watch the fact-checker catch it
streamlit run app.py                            # dashboard (best for presenting)
pytest -q                                       # 4 tests
```

Approved briefs go to `outputs/reports/`, rejected ones to `outputs/drafts/`. Every agent step is appended to `outputs/run_log.csv`.

## Real LLM mode (CrewAI)

```bash
pip install -r requirements-crew.txt
cp .env.example .env        # add your API key, optionally set MODEL
python main.py "Agentic AI" --mode crew
```

Same four roles, built as CrewAI agents and tasks (`src/crew_pipeline.py`). Demo mode uses rule-based agents so the
demo is free, offline and repeatable. Crew mode needs a live API key and was not run during packaging.

## Presentation flow (about 5 minutes)

1. **Problem:** AI-written research can contain unsupported claims. Show the pipeline diagram.
2. **Demo 1:** Dashboard, topic "Agentic AI", Run. Walk through the agent trace table and the cited brief.
3. **Demo 2:** Tick "Inject fake claim", Run again. The fact-checker marks it unverified and the brief leaves it out.
4. **Human gate:** Click Approve (saved to reports) or Reject (saved to drafts).
5. **Analytics tab:** 4,500 logged agent runs (`data/dataset_additional.csv`): tokens, time, quality and error rate by agent.
6. **No-code vs code (P_170 theme):** The knowledge base doc `no_code_agents.md` covers when to use Copilot Studio (simple FAQ/approvals)
   versus Python/CrewAI (multi-step logic, testing, custom tools). This project is the code side.

## Project layout

```
main.py                 CLI
app.py                  Streamlit dashboard
src/kb.py               local knowledge base search and claim-support check
src/demo_pipeline.py    offline 4-agent pipeline + human approval gate
src/crew_pipeline.py    CrewAI version of the same agents
src/logger.py           run log (same columns as the dataset)
data/knowledge_base/    6 short sample documents (add your own .md files: first line "# Title", then one fact per "- " line)
data/dataset_additional.csv   synthetic agent-run logs for the analytics tab
tests/                  pytest checks
Dockerfile              docker build -t research-crew . && docker run -p 8501:8501 research-crew
```

## Limits (be upfront about these in the presentation)

- The sample knowledge base is small, so topics outside it return "no sources" by design.
- Demo-mode fact-checking is keyword overlap with the cited source, not semantic understanding. Crew mode uses an LLM checker.
- Token counts in demo mode are estimates (words x 1.3). Quality scores are heuristics.
- `data/dataset_additional.csv` is synthetic and only feeds the analytics view.

## Ethics

Multi-agent systems can spread misinformation. This project reduces that risk with source-grounded claims, a dedicated
fact-checker, an audit log of every step, and mandatory human approval before publishing.
