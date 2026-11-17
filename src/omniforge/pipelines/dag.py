"""Directed Acyclic Graph (DAG) representation with cycle detection, topological ordering, and Mermaid export."""

from __future__ import annotations

from collections import deque
from typing import Any

from omniforge.core.exceptions import PipelineError
from omniforge.pipelines.base import PipelineStep


class PipelineDAG:
    """Validated DAG of :class:`PipelineStep` nodes supporting topological sorting and visual export."""

    def __init__(self, name: str = "uap_pipeline") -> None:
        self.name = name
        self._steps: dict[str, PipelineStep] = {}

    def add_step(self, step: PipelineStep) -> PipelineDAG:
        """Add a step node to the DAG."""
        if step.step_id in self._steps:
            raise PipelineError(f"Duplicate step_id '{step.step_id}' in pipeline '{self.name}'.")
        self._steps[step.step_id] = step
        return self

    def get_step(self, step_id: str) -> PipelineStep:
        """Retrieve a step node by ID."""
        if step_id not in self._steps:
            raise PipelineError(f"Step '{step_id}' not found in pipeline '{self.name}'.")
        return self._steps[step_id]

    @property
    def steps(self) -> dict[str, PipelineStep]:
        """Return mapping of step_id to :class:`PipelineStep`."""
        return dict(self._steps)

    def validate(self) -> None:
        """Validate dependency existence and verify graph acyclicity."""
        for sid, step in self._steps.items():
            for dep in step.depends_on:
                if dep not in self._steps:
                    raise PipelineError(
                        f"Step '{sid}' depends on unknown step '{dep}'.",
                        suggestions=["Add upstream step before compiling/executing the DAG."],
                    )
        # Topological sort raises if a cycle exists
        self.topological_order()

    def topological_order(self) -> list[str]:
        """Return valid Kahn topological execution order or raise :class:`PipelineError` on cycles."""
        in_degree: dict[str, int] = {sid: 0 for sid in self._steps}
        children: dict[str, list[str]] = {sid: [] for sid in self._steps}

        for sid, step in self._steps.items():
            for dep in step.depends_on:
                if dep not in self._steps:
                    raise PipelineError(f"Missing dependency '{dep}' for step '{sid}'.")
                in_degree[sid] += 1
                children[dep].append(sid)

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for nxt in children[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self._steps):
            raise PipelineError(
                f"Cycle detected in pipeline DAG '{self.name}'.",
                suggestions=["Remove circular depends_on references between pipeline steps."],
            )
        return order

    def execution_levels(self) -> list[list[str]]:
        """Group steps into parallelizable execution waves."""
        order = self.topological_order()
        depth: dict[str, int] = {}
        for sid in order:
            deps = self._steps[sid].depends_on
            depth[sid] = 0 if not deps else (1 + max(depth[d] for d in deps))
        max_d = max(depth.values(), default=-1)
        return [[sid for sid in order if depth[sid] == lvl] for lvl in range(max_d + 1)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize DAG structure (compatible with visual drag-and-drop Pipeline Designer)."""
        return {
            "name": self.name,
            "steps": [self._steps[sid].to_spec() for sid in self.topological_order()],
        }

    def to_mermaid(self) -> str:
        """Render the DAG as a Mermaid flowchart string."""
        lines = ["flowchart LR"]
        for sid in self.topological_order():
            step = self._steps[sid]
            safe_label = f"{sid}[{sid}<br/>{step.step_type}]"
            lines.append(f"    {safe_label}")
            for dep in step.depends_on:
                lines.append(f"    {dep} --> {sid}")
        return "\n".join(lines)
