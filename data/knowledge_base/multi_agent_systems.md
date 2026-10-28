# Multi-Agent Systems
- A multi-agent system splits a complex task between several agents that each have a defined role and goal.
- Frameworks such as CrewAI organise agents into a crew that works through tasks in sequence or in a hierarchy.
- Dividing work by role keeps each prompt focused and makes failures easier to locate than in a single large prompt.
- Agents in a crew pass outputs to each other, so an error made early can propagate to later agents.
- Adding a dedicated reviewer or fact-checking agent reduces the chance that an unsupported claim reaches the final output.
- Multi-agent runs usually consume more tokens and time than a single call, so cost monitoring matters.
