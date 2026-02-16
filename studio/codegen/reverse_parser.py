"""Reverse parser: Python AST -> WorkflowIR.

Uses heuristic pattern matching to classify decorated state methods
back into node types and reconstruct a WorkflowIR.
"""

from __future__ import annotations

import ast
import re

from .ir import (
    AgentConfig,
    ConditionalConfig,
    Edge,
    FanOutConfig,
    FunctionConfig,
    LLMConfig,
    MemoryConfig,
    MergeConfig,
    Node,
    NodeConfig,
    NodeType,
    OutputConfig,
    Position,
    StoreConfig,
    ToolConfig,
    WorkflowIR,
    WorkflowInput,
    WorkflowMetadata,
    WorkflowOutput,
)


class ReverseParser:
    """Parse PuffinFlow agent source code back into a WorkflowIR."""

    def parse(self, source: str) -> WorkflowIR:
        """Parse a Python source string into a WorkflowIR."""
        tree = ast.parse(source)
        class_node = self._find_agent_class(tree)
        if class_node is None:
            raise ValueError("No Agent subclass found in source")
        return self._build_ir(class_node, source)

    def _find_agent_class(self, tree: ast.Module) -> ast.ClassDef | None:
        """Find the first class that inherits from Agent."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "Agent":
                        return node
        return None

    def _is_state_method(self, func_node: ast.AsyncFunctionDef) -> bool:
        """Check if a function has a @state() decorator."""
        for dec in func_node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "state":
                return True
            if isinstance(dec, ast.Name) and dec.id == "state":
                return True
        return False

    def _classify_method(self, func_node: ast.AsyncFunctionDef) -> NodeType:
        """Classify a state method into a NodeType using heuristics."""
        source = ast.dump(func_node)

        # Check patterns in order of specificity
        if "Send(" in source or "Send" in source:
            # Look for actual Send usage (fan-out pattern)
            for node in ast.walk(func_node):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == "Send":
                        return NodeType.FAN_OUT

        if "ctx.store." in source or "store.put" in source or "store.get" in source:
            return NodeType.MEMORY

        # LLM patterns
        if any(
            pattern in source
            for pattern in ["llm_client", "openai", "anthropic", "llm.call"]
        ):
            return NodeType.LLM

        # Pure conditional: only if/return pattern
        if self._is_pure_conditional(func_node):
            return NodeType.CONDITIONAL

        # Output: only ctx.set_output calls followed by return None
        if self._is_output_only(func_node):
            return NodeType.OUTPUT

        # Merge: very simple pass-through
        if self._is_merge_pattern(func_node):
            return NodeType.MERGE

        # Default
        return NodeType.FUNCTION

    def _is_pure_conditional(self, func_node: ast.AsyncFunctionDef) -> bool:
        """Check if the method is purely an if/return branching pattern."""
        body = func_node.body
        if not body:
            return False

        # Should have an If statement and Return statements
        has_if = False
        for stmt in body:
            if isinstance(stmt, ast.If):
                has_if = True
                # The if body should contain a return
                if not any(isinstance(s, ast.Return) for s in stmt.body):
                    return False
        return has_if and len(body) <= 3

    def _is_output_only(self, func_node: ast.AsyncFunctionDef) -> bool:
        """Check if the method only does ctx.set_output() calls."""
        for stmt in func_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute) and func.attr == "set_output":
                    continue
            elif isinstance(stmt, ast.Return):
                continue
            else:
                return False
        return True

    def _is_merge_pattern(self, func_node: ast.AsyncFunctionDef) -> bool:
        """Check if method is a simple pass-through (merge node)."""
        body = [
            s
            for s in func_node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        return len(body) == 1 and isinstance(body[0], ast.Return)

    def _extract_edges(
        self, func_node: ast.AsyncFunctionDef, method_name: str
    ) -> list[Edge]:
        """Extract edges from return statements in the method."""
        edges: list[Edge] = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                val = node.value

                # String literal return -> edge to that state
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    edges.append(Edge(from_node=method_name, to_node=val.value))

                # None return -> terminal (no edge)
                elif isinstance(val, ast.Constant) and val.value is None:
                    pass

                # Command(goto=...) pattern
                elif isinstance(val, ast.Call):
                    func = val.func
                    if isinstance(func, ast.Name) and func.id == "Command":
                        for kw in val.keywords:
                            if kw.arg == "goto" and isinstance(
                                kw.value, ast.Constant
                            ):
                                edges.append(
                                    Edge(
                                        from_node=method_name,
                                        to_node=kw.value.value,
                                    )
                                )

                # List comprehension with Send -> fan-out
                elif isinstance(val, ast.ListComp):
                    pass  # Fan-out edges are implicit

        # Handle if/else labeled edges
        for stmt in func_node.body:
            if isinstance(stmt, ast.If):
                for sub in stmt.body:
                    if isinstance(sub, ast.Return) and isinstance(
                        sub.value, ast.Constant
                    ) and isinstance(sub.value.value, str):
                        # Mark as true edge
                        for e in edges:
                            if e.to_node == sub.value.value and e.label is None:
                                e.label = "true"
                if stmt.orelse:
                    for sub in stmt.orelse:
                        if isinstance(sub, ast.Return) and isinstance(
                            sub.value, ast.Constant
                        ) and isinstance(sub.value.value, str):
                            for e in edges:
                                if (
                                    e.to_node == sub.value.value
                                    and e.label is None
                                ):
                                    e.label = "false"

        return edges

    def _extract_llm_config(
        self, func_node: ast.AsyncFunctionDef
    ) -> LLMConfig | None:
        """Try to extract LLM configuration from a method."""
        model = "gpt-4"
        output_key = "response"

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Look for llm_client.call(...)
                for kw in node.keywords:
                    if kw.arg == "model" and isinstance(kw.value, ast.Constant):
                        model = kw.value.value
                    if kw.arg == "temperature" and isinstance(
                        kw.value, ast.Constant
                    ):
                        pass

            # Look for ctx.set_variable("key", response) to find output_key
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_variable" and len(node.args) >= 1 and isinstance(node.args[0], ast.Constant):
                output_key = node.args[0].value

        return LLMConfig(model=model, output_key=output_key)

    def _extract_conditional_config(
        self, func_node: ast.AsyncFunctionDef
    ) -> ConditionalConfig | None:
        """Extract conditional branching config."""
        for stmt in func_node.body:
            if isinstance(stmt, ast.If):
                condition = ast.unparse(stmt.test)
                true_target = ""
                false_target = ""

                for sub in stmt.body:
                    if isinstance(sub, ast.Return) and isinstance(
                        sub.value, ast.Constant
                    ):
                        true_target = sub.value.value

                # False target from orelse or next statement
                if stmt.orelse:
                    for sub in stmt.orelse:
                        if isinstance(sub, ast.Return) and isinstance(
                            sub.value, ast.Constant
                        ):
                            false_target = sub.value.value

                # Check statements after the if for the false return
                idx = func_node.body.index(stmt)
                for sub in func_node.body[idx + 1 :]:
                    if isinstance(sub, ast.Return) and isinstance(
                        sub.value, ast.Constant
                    ):
                        false_target = sub.value.value
                        break

                if true_target and false_target:
                    return ConditionalConfig(
                        condition=condition,
                        true_target=true_target,
                        false_target=false_target,
                    )
        return None

    def _extract_output_config(
        self, func_node: ast.AsyncFunctionDef
    ) -> OutputConfig | None:
        """Extract output mappings from set_output calls."""
        mappings: dict[str, str] = {}
        for stmt in func_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute) and func.attr == "set_output":
                    args = stmt.value.args
                    if len(args) >= 2 and isinstance(args[0], ast.Constant):
                        output_name = args[0].value
                        # Try to extract the variable key from
                        # ctx.get_variable("key")
                        if isinstance(args[1], ast.Call):
                            inner = args[1]
                            if (
                                isinstance(inner.func, ast.Attribute)
                                and inner.func.attr == "get_variable"
                                and inner.args
                                and isinstance(inner.args[0], ast.Constant)
                            ):
                                mappings[output_name] = inner.args[0].value
                            else:
                                mappings[output_name] = output_name
                        else:
                            mappings[output_name] = output_name
        if mappings:
            return OutputConfig(mappings=mappings)
        return None

    def _extract_init_info(
        self, class_node: ast.ClassDef
    ) -> tuple[str, str, StoreConfig | None]:
        """Extract agent name and store config from __init__."""
        agent_name = class_node.name.lower()
        class_name = class_node.name
        store_config: StoreConfig | None = None

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in item.body:
                    if isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, ast.Call
                    ):
                        call = stmt.value
                        # super().__init__("name", ...) pattern
                        if isinstance(call.func, ast.Attribute) and call.func.attr == "__init__":
                            if call.args and isinstance(
                                call.args[0], ast.Constant
                            ):
                                agent_name = call.args[0].value
                            for kw in call.keywords:
                                if kw.arg == "store":
                                    store_config = StoreConfig(type="memory")

        return agent_name, class_name, store_config

    def _build_ir(self, class_node: ast.ClassDef, source: str) -> WorkflowIR:
        """Build a WorkflowIR from the parsed class."""
        agent_name, class_name, store_config = self._extract_init_info(class_node)

        nodes: list[Node] = []
        all_edges: list[Edge] = []
        y_offset = 0.0

        for item in class_node.body:
            if not isinstance(item, ast.AsyncFunctionDef):
                continue
            if not self._is_state_method(item):
                continue

            method_name = item.name
            node_type = self._classify_method(item)

            # Build config based on type
            config = NodeConfig()
            if node_type == NodeType.LLM:
                config.llm = self._extract_llm_config(item)
            elif node_type == NodeType.CONDITIONAL:
                config.conditional = self._extract_conditional_config(item)
            elif node_type == NodeType.OUTPUT:
                config.output = self._extract_output_config(item)
            elif node_type == NodeType.FUNCTION:
                config.function = FunctionConfig()

            node = Node(
                id=method_name,
                type=node_type,
                position=Position(x=0.0, y=y_offset),
                config=config,
            )
            nodes.append(node)
            y_offset += 100.0

            # Extract edges
            edges = self._extract_edges(item, method_name)
            all_edges.extend(edges)

        return WorkflowIR(
            metadata=WorkflowMetadata(name=agent_name),
            agent=AgentConfig(
                name=agent_name,
                class_name=class_name,
                store=store_config,
            ),
            nodes=nodes,
            edges=all_edges,
        )
