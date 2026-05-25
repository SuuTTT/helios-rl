# Deep Research Request Template

Use this prompt in a deep-research web app before creating the local benchmark.

Topic:
- <research idea>

Goal:
- Identify the current SOTA, benchmark tasks, metrics, official repos, and
  reproduction pitfalls.

Please produce:
1. A short problem statement.
2. A table of top methods with paper, year, venue, benchmark, metric, result,
   compute budget, and code availability.
3. The official benchmark definition and evaluation protocol.
4. The strongest open-source implementation for each baseline.
5. Known reproduction issues and dependency constraints.
6. Dataset/task download requirements.
7. What result would count as beating SOTA fairly.
8. Suggested ablations and controls.
9. BibTeX entries for cited papers.
10. A prioritized plan for building a local testbed.

Constraints:
- Clearly separate reported results from your inference.
- Prefer official sources and papers over secondary summaries.
- Include links to code and benchmark docs.
- Flag benchmark inconsistencies or unfair comparisons.

