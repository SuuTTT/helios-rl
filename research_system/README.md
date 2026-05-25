# Research System Framework

This directory contains generic templates for turning the current TD-MPC-Glass
workflow into a reusable research operating system.

Start a project:

```bash
mkdir -p research/my_project/{deep_research,benchmark,probes,analysis,blog,paper}
cp research_system/templates/project.yaml research/my_project/project.yaml
cp research_system/templates/deep_research_request.md research/my_project/deep_research/request.md
cp research_system/templates/blog_post.md research/my_project/blog/YYYY-MM-DD-milestone.md
cp research_system/templates/paper_main.tex research/my_project/paper/main.tex
```

Capture ideas:

```bash
/root/venv/bin/python3 scripts/idea_queue.py add \
  --title "..." \
  --goal "..." \
  --hypothesis "..." \
  --metric "..." \
  --tags "my_project" \
  --priority 5
```

Read the full design:

```text
docs/research_system/research_workflow_platform.md
```

