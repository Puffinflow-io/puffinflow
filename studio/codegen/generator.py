"""Main code generator: WorkflowIR -> Python source code."""

from __future__ import annotations

from collections import defaultdict, deque

from .compilers import compile_node
from .formatter import format_code
from .ir import Node, NodeType, WorkflowIR


class CodeGenerator:
    """Generate executable PuffinFlow agent code from a WorkflowIR."""

    def __init__(self, ir: WorkflowIR) -> None:
        self.ir = ir

    def generate(self) -> str:
        """Generate the full Python module source."""
        sections = [
            self._gen_imports(),
            self._gen_class(),
            self._gen_main(),
        ]
        raw = "\n\n".join(sections)
        return format_code(raw)

    # ------------------------------------------------------------------
    # Import generation
    # ------------------------------------------------------------------

    def _gen_imports(self) -> str:
        imports = ["from puffinflow import Agent, state"]

        has_fan_out = any(n.type == NodeType.FAN_OUT for n in self.ir.nodes)
        has_memory = any(n.type == NodeType.MEMORY for n in self.ir.nodes)
        has_command = any(n.type == NodeType.MERGE for n in self.ir.nodes)

        extra: list[str] = []
        if has_fan_out:
            extra.append("Send")
        if has_command:
            extra.append("Command")
        if self.ir.agent.store or has_memory:
            extra.append("MemoryStore")

        if extra:
            imports[0] = f"from puffinflow import Agent, state, {', '.join(extra)}"

        # Reducer imports
        reducer_types = {r.type for r in self.ir.agent.reducers}
        reducer_imports: list[str] = []
        if "append" in reducer_types:
            reducer_imports.append("append_reducer")
        if "replace" in reducer_types:
            reducer_imports.append("replace_reducer")
        if "add" in reducer_types:
            reducer_imports.append("add_reducer")
        if reducer_imports:
            imports.append(f"from puffinflow import {', '.join(reducer_imports)}")

        return "\n".join(imports)

    # ------------------------------------------------------------------
    # Class generation
    # ------------------------------------------------------------------

    def _gen_class(self) -> str:
        ir = self.ir
        class_name = ir.agent.class_name
        lines: list[str] = []

        lines.append(f"class {class_name}(Agent):")
        lines.append(f'    """Auto-generated agent: {ir.metadata.name}."""')
        lines.append("")

        # __init__
        lines.append("    def __init__(self):")
        store_arg = ""
        if ir.agent.store:
            if ir.agent.store.type == "memory":
                store_arg = ", store=MemoryStore()"
            else:
                store_arg = ", store=MemoryStore()"
        lines.append(f'        super().__init__("{ir.agent.name}"{store_arg})')

        # Register reducers
        for reducer in ir.agent.reducers:
            if reducer.type == "append":
                lines.append(
                    f'        self.add_reducer("{reducer.key}", append_reducer)'
                )
            elif reducer.type == "replace":
                lines.append(
                    f'        self.add_reducer("{reducer.key}", replace_reducer)'
                )
            elif reducer.type == "add":
                lines.append(f'        self.add_reducer("{reducer.key}", add_reducer)')
            elif reducer.type == "custom" and reducer.module:
                mod, func = reducer.module.rsplit(".", 1)
                lines.append(f"        from {mod} import {func}")
                lines.append(f'        self.add_reducer("{reducer.key}", {func})')

        lines.append("")

        # State methods in topological order
        sorted_nodes = self._topological_sort()
        for node in sorted_nodes:
            # Skip structural-only node types that are handled elsewhere
            if node.type == NodeType.SUBGRAPH:
                continue

            # Build decorator kwargs
            decorator_args: list[str] = []
            if node.resources:
                if node.resources.cpu is not None:
                    decorator_args.append(f"cpu={node.resources.cpu}")
                if node.resources.memory is not None:
                    decorator_args.append(f"memory={node.resources.memory}")
                if node.resources.gpu is not None:
                    decorator_args.append(f"gpu={node.resources.gpu}")
                if node.resources.timeout is not None:
                    decorator_args.append(f"timeout={node.resources.timeout}")

            if decorator_args:
                lines.append(f"    @state({', '.join(decorator_args)})")
            else:
                lines.append("    @state()")

            method_name = node.id.replace("-", "_").replace(" ", "_")
            lines.append(f"    async def {method_name}(self, ctx):")

            body_lines = compile_node(node, self.ir.edges, self.ir.nodes)
            if body_lines:
                for bl in body_lines:
                    lines.append("    " + bl)
            else:
                lines.append("        return None")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main block generation
    # ------------------------------------------------------------------

    def _gen_main(self) -> str:
        ir = self.ir
        class_name = ir.agent.class_name

        lines = [
            'if __name__ == "__main__":',
            "    import asyncio",
            "",
            f"    agent = {class_name}()",
            "    result = asyncio.run(agent.run())",
            "    print(result.outputs)",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self) -> list[Node]:
        """Return nodes in topological order based on edges."""
        node_map = {n.id: n for n in self.ir.nodes}
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {n.id: 0 for n in self.ir.nodes}

        for edge in self.ir.edges:
            adj[edge.from_node].append(edge.to_node)
            if edge.to_node in in_degree:
                in_degree[edge.to_node] += 1

        queue: deque[str] = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)

        sorted_ids: list[str] = []
        while queue:
            nid = queue.popleft()
            sorted_ids.append(nid)
            for neighbor in adj[nid]:
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Append any remaining nodes not reachable via edges
        seen = set(sorted_ids)
        for n in self.ir.nodes:
            if n.id not in seen:
                sorted_ids.append(n.id)

        return [node_map[nid] for nid in sorted_ids if nid in node_map]
