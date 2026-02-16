"""Node compilers: one function per NodeType.

Each compiler takes (node, edges, all_nodes) and returns a list of
indented Python source lines for the method body.
"""

from __future__ import annotations

import re

from .ir import Edge, Node, NodeType


def _find_next_state(
    node: Node, edges: list[Edge], all_nodes: list[Node]
) -> str | None:
    """Find the single outgoing target state for a node."""
    for edge in edges:
        if edge.from_node == node.id:
            return edge.to_node
    return None


def _template_vars(text: str) -> list[str]:
    """Extract {{var}} placeholders from a template string."""
    return re.findall(r"\{\{(\w+)\}\}", text)


def compile_llm(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate LLM call code."""
    cfg = node.config.llm
    if cfg is None:
        return ["    return None"]
    lines: list[str] = []

    # Extract template variables from user_prompt
    template_vars = _template_vars(cfg.user_prompt)
    for var in template_vars:
        lines.append(f'    {var} = ctx.get_variable("{var}")')

    # Build prompt
    if template_vars:
        prompt_str = cfg.user_prompt
        for var in template_vars:
            prompt_str = prompt_str.replace("{{" + var + "}}", "{" + var + "}")
        lines.append(f'    prompt = f"{prompt_str}"')
    else:
        lines.append(f"    prompt = {cfg.user_prompt!r}")

    # System prompt
    if cfg.system_prompt:
        lines.append(f"    system_prompt = {cfg.system_prompt!r}")
        lines.append(
            f'    response = await llm_client.call('
            f'prompt, system=system_prompt, '
            f'model="{cfg.model}", '
            f'temperature={cfg.temperature}'
            f'{f", max_tokens={cfg.max_tokens}" if cfg.max_tokens else ""})'
        )
    else:
        lines.append(
            f'    response = await llm_client.call('
            f'prompt, '
            f'model="{cfg.model}", '
            f'temperature={cfg.temperature}'
            f'{f", max_tokens={cfg.max_tokens}" if cfg.max_tokens else ""})'
        )

    lines.append(f'    ctx.set_variable("{cfg.output_key}", response)')

    next_state = _find_next_state(node, edges, all_nodes)
    if next_state:
        lines.append(f'    return "{next_state}"')
    else:
        lines.append("    return None")

    return lines


def compile_function(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate function call or inline code."""
    cfg = node.config.function
    if cfg is None:
        return ["    return None"]
    lines: list[str] = []

    # Read input variables
    for key in cfg.input_keys:
        lines.append(f'    {key} = ctx.get_variable("{key}")')

    if cfg.code:
        # Insert inline code, indented at method level
        for code_line in cfg.code.splitlines():
            lines.append(f"    {code_line}")
    elif cfg.module:
        # Import and call external function
        parts = cfg.module.rsplit(".", 1)
        if len(parts) == 2:
            mod_path, func_name = parts
            lines.append(f"    from {mod_path} import {func_name}")
            if cfg.input_keys:
                args = ", ".join(cfg.input_keys)
                lines.append(f"    result = {func_name}({args})")
            else:
                lines.append(f"    result = {func_name}()")
        else:
            lines.append(f"    result = {cfg.module}()")

    # Set output
    if cfg.output_key and (cfg.code or cfg.module):
        lines.append(f'    ctx.set_variable("{cfg.output_key}", result)')

    next_state = _find_next_state(node, edges, all_nodes)
    if next_state:
        lines.append(f'    return "{next_state}"')
    else:
        lines.append("    return None")

    return lines


def compile_conditional(
    node: Node, edges: list[Edge], all_nodes: list[Node]
) -> list[str]:
    """Generate if/else branching."""
    cfg = node.config.conditional
    if cfg is None:
        return ["    return None"]
    lines = [
        f"    if {cfg.condition}:",
        f'        return "{cfg.true_target}"',
        f'    return "{cfg.false_target}"',
    ]
    return lines


def compile_input(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate input variable initialization."""
    cfg = node.config.input
    if cfg is None:
        return ["    return None"]
    lines: list[str] = []

    for var_def in cfg.variables:
        name = var_def.get("name", "")
        default = var_def.get("default")
        if default is not None:
            lines.append(
                f'    ctx.set_variable("{name}", '
                f'ctx.get_variable("{name}", {default!r}))'
            )
        else:
            lines.append(f'    ctx.set_variable("{name}", ctx.get_variable("{name}"))')

    next_state = _find_next_state(node, edges, all_nodes)
    if next_state:
        lines.append(f'    return "{next_state}"')
    else:
        lines.append("    return None")

    return lines


def compile_output(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate output mapping."""
    cfg = node.config.output
    if cfg is None:
        return ["    return None"]
    lines: list[str] = []

    for output_name, var_key in cfg.mappings.items():
        lines.append(
            f'    ctx.set_output("{output_name}", ctx.get_variable("{var_key}"))'
        )

    lines.append("    return None")
    return lines


def compile_tool(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate tool invocation."""
    cfg = node.config.tool
    if cfg is None:
        return ["    return None"]
    lines: list[str] = []

    # Import tool if module specified
    if cfg.tool_module:
        parts = cfg.tool_module.rsplit(".", 1)
        if len(parts) == 2:
            mod_path, func_name = parts
            lines.append(f"    from {mod_path} import {func_name}")
            tool_ref = func_name
        else:
            tool_ref = cfg.tool_module
    else:
        tool_ref = cfg.tool_name

    # Build parameters
    if cfg.parameters:
        param_parts = []
        for k, v in cfg.parameters.items():
            if isinstance(v, str) and v.startswith("ctx."):
                param_parts.append(f"{k}={v}")
            else:
                param_parts.append(f"{k}={v!r}")
        params = ", ".join(param_parts)
        lines.append(f"    result = await {tool_ref}({params})")
    else:
        lines.append(f"    result = await {tool_ref}()")

    lines.append(f'    ctx.set_variable("{cfg.output_key}", result)')

    next_state = _find_next_state(node, edges, all_nodes)
    if next_state:
        lines.append(f'    return "{next_state}"')
    else:
        lines.append("    return None")

    return lines


def compile_memory(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate memory store operations."""
    cfg = node.config.memory
    if cfg is None:
        return ["    return None"]
    lines: list[str] = []

    ns_tuple = repr(tuple(cfg.namespace))
    lines.append(f"    namespace = {ns_tuple}")

    if cfg.operation == "put":
        key_expr = f'ctx.get_variable("{cfg.key}")' if cfg.key else '""'
        val_expr = f'ctx.get_variable("{cfg.value_key}")' if cfg.value_key else "None"
        lines.append(f"    key = {key_expr}")
        lines.append(f"    value = {val_expr}")
        lines.append("    await ctx.store.put(namespace, key, value)")
    elif cfg.operation == "get":
        key_expr = f'ctx.get_variable("{cfg.key}")' if cfg.key else '""'
        lines.append(f"    key = {key_expr}")
        lines.append("    item = await ctx.store.get(namespace, key)")
        lines.append("    result = item.value if item else None")
        lines.append(f'    ctx.set_variable("{cfg.output_key}", result)')
    elif cfg.operation == "delete":
        key_expr = f'ctx.get_variable("{cfg.key}")' if cfg.key else '""'
        lines.append(f"    key = {key_expr}")
        lines.append("    result = await ctx.store.delete(namespace, key)")
        lines.append(f'    ctx.set_variable("{cfg.output_key}", result)')
    elif cfg.operation == "list":
        lines.append(f"    items = await ctx.store.list(namespace, limit={cfg.limit})")
        lines.append("    result = [item.value for item in items]")
        lines.append(f'    ctx.set_variable("{cfg.output_key}", result)')
    elif cfg.operation == "search":
        query_expr = f'ctx.get_variable("{cfg.query}")' if cfg.query else '""'
        lines.append(f"    query = {query_expr}")
        lines.append(
            f"    items = await ctx.store.search(namespace, query=query, "
            f"limit={cfg.limit})"
        )
        lines.append("    result = [item.value for item in items]")
        lines.append(f'    ctx.set_variable("{cfg.output_key}", result)')

    next_state = _find_next_state(node, edges, all_nodes)
    if next_state:
        lines.append(f'    return "{next_state}"')
    else:
        lines.append("    return None")

    return lines


def compile_fan_out(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate fan-out dispatch."""
    cfg = node.config.fan_out
    if cfg is None:
        return ["    return None"]
    lines = [
        f'    items = ctx.get_variable("{cfg.items_key}")',
        f'    return [Send("{cfg.target_state}", '
        f'{{"{cfg.item_variable}": item}}) for item in items]',
    ]
    return lines


def compile_merge(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Generate merge passthrough.

    Merge nodes are mostly handled by reducers declared on the agent;
    the state itself just passes through to the next node.
    """
    lines: list[str] = []
    next_state = _find_next_state(node, edges, all_nodes)
    if next_state:
        lines.append(f'    return "{next_state}"')
    else:
        lines.append("    return None")
    return lines


def compile_node(node: Node, edges: list[Edge], all_nodes: list[Node]) -> list[str]:
    """Main dispatcher: compile a node into method body lines."""
    compilers = {
        NodeType.LLM: compile_llm,
        NodeType.FUNCTION: compile_function,
        NodeType.CONDITIONAL: compile_conditional,
        NodeType.INPUT: compile_input,
        NodeType.OUTPUT: compile_output,
        NodeType.TOOL: compile_tool,
        NodeType.MEMORY: compile_memory,
        NodeType.FAN_OUT: compile_fan_out,
        NodeType.MERGE: compile_merge,
    }
    compiler = compilers.get(node.type)
    if compiler:
        return compiler(node, edges, all_nodes)
    return []
