"""Project template scaffolding for PuffinFlow Studio."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------
    "basic": {
        "workflows/main.yaml": """\
version: "1.0"
metadata:
  name: "{name}"
  description: "Basic single-agent workflow"
  author: ""
  tags: []
agent:
  name: "{name_lower}"
  class_name: "{class_name}"
  max_concurrent: 5
inputs:
  - name: query
    type: str
outputs:
  - name: answer
    type: str
nodes:
  - id: process
    type: llm
    config:
      llm:
        model: gpt-4
        user_prompt: "Answer the following question: {{{{query}}}}"
        output_key: answer
  - id: done
    type: output
    config:
      output:
        mappings:
          answer: answer
edges:
  - from_node: process
    to_node: done
""",
        "evals/main_eval.yaml": """\
name: "{name}-eval"
workflow: "workflows/main.yaml"
scoring:
  default_scorer: contains
  threshold: 0.7
cases:
  - name: basic_test
    input:
      query: "What is 2 + 2?"
    expected:
      answer_contains: ["4"]
""",
    },
    # ------------------------------------------------------------------
    "research": {
        "workflows/main.yaml": """\
version: "1.0"
metadata:
  name: "{name}"
  description: "Multi-step research agent with memory"
  author: ""
  tags: [research, memory]
agent:
  name: "{name_lower}"
  class_name: "{class_name}"
  max_concurrent: 5
  store:
    type: memory
inputs:
  - name: topic
    type: str
outputs:
  - name: summary
    type: str
nodes:
  - id: research
    type: llm
    config:
      llm:
        model: gpt-4
        user_prompt: "Research the following topic in depth: {{{{topic}}}}"
        output_key: findings
  - id: check_quality
    type: conditional
    config:
      conditional:
        condition: "len(ctx.get_variable('findings', '')) > 100"
        true_target: summarize
        false_target: research
  - id: store_findings
    type: memory
    config:
      memory:
        operation: put
        namespace: [research]
        key: findings
        value_key: findings
        output_key: stored
  - id: summarize
    type: llm
    config:
      llm:
        model: gpt-4
        user_prompt: "Create a concise summary of: {{{{findings}}}}"
        output_key: summary
  - id: done
    type: output
    config:
      output:
        mappings:
          summary: summary
edges:
  - from_node: research
    to_node: check_quality
  - from_node: check_quality
    to_node: summarize
    label: "true"
  - from_node: check_quality
    to_node: research
    label: "false"
  - from_node: summarize
    to_node: store_findings
  - from_node: store_findings
    to_node: done
""",
        "evals/main_eval.yaml": """\
name: "{name}-eval"
workflow: "workflows/main.yaml"
scoring:
  default_scorer: contains
  threshold: 0.5
cases:
  - name: quantum_computing
    input:
      topic: "quantum computing"
    expected:
      summary_contains: ["qubit", "quantum"]
  - name: machine_learning
    input:
      topic: "machine learning"
    expected:
      summary_contains: ["model", "training"]
""",
    },
    # ------------------------------------------------------------------
    "pipeline": {
        "workflows/main.yaml": """\
version: "1.0"
metadata:
  name: "{name}"
  description: "Pipeline with fan-out / merge pattern"
  author: ""
  tags: [pipeline, parallel]
agent:
  name: "{name_lower}"
  class_name: "{class_name}"
  max_concurrent: 10
  reducers:
    - key: results
      type: append
inputs:
  - name: items
    type: list
outputs:
  - name: results
    type: list
nodes:
  - id: scatter
    type: fan_out
    config:
      fan_out:
        items_key: items
        target_state: process_item
        item_variable: item
  - id: process_item
    type: function
    config:
      function:
        code: |
          item = ctx.get_variable("item")
          result = f"processed: {{item}}"
          ctx.set_variable("result", result)
        output_key: result
  - id: gather
    type: merge
    config:
      merge:
        reducer_key: results
        strategy: append
  - id: done
    type: output
    config:
      output:
        mappings:
          results: results
edges:
  - from_node: scatter
    to_node: process_item
  - from_node: process_item
    to_node: gather
  - from_node: gather
    to_node: done
""",
        "evals/main_eval.yaml": """\
name: "{name}-eval"
workflow: "workflows/main.yaml"
scoring:
  default_scorer: exact_match
  threshold: 1.0
cases:
  - name: basic_pipeline
    input:
      items: ["a", "b", "c"]
    expected:
      results: ["processed: a", "processed: b", "processed: c"]
""",
    },
}

GITIGNORE = """\
__pycache__/
*.py[cod]
.env
*.db
.venv/
node_modules/
.next/
dist/
"""


def scaffold_project(
    name: str,
    template: str = "basic",
    output_dir: str | Path = ".",
) -> Path:
    """Create a new PuffinFlow project from a template.

    Returns the project root directory.
    """
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template '{template}'. Choose from: {list(TEMPLATES)}")

    root = Path(output_dir) / name
    root.mkdir(parents=True, exist_ok=True)

    # Formatting context
    class_name = name.replace("-", " ").replace("_", " ").title().replace(" ", "")
    fmt = {
        "name": name,
        "name_lower": name.lower().replace("-", "_"),
        "class_name": class_name,
    }

    files = TEMPLATES[template]
    for rel_path, content in files.items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content.format(**fmt))

    # Deploy dir (empty placeholder)
    (root / "deploy").mkdir(exist_ok=True)

    # .gitignore
    (root / ".gitignore").write_text(GITIGNORE)

    return root
