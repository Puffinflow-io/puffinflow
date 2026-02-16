from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    LLM = "llm"
    FUNCTION = "function"
    CONDITIONAL = "conditional"
    INPUT = "input"
    OUTPUT = "output"
    SUBGRAPH = "subgraph"
    TOOL = "tool"
    MEMORY = "memory"
    FAN_OUT = "fan_out"
    MERGE = "merge"


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class LLMConfig(BaseModel):
    model: str = "gpt-4"
    system_prompt: str | None = None
    user_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    output_key: str = "response"


class FunctionConfig(BaseModel):
    code: str | None = None  # Inline Python code
    module: str | None = None  # External module.function path
    input_keys: list[str] = Field(default_factory=list)
    output_key: str | None = None


class ConditionalConfig(BaseModel):
    condition: str  # Python expression using ctx.get_variable()
    true_target: str
    false_target: str


class InputConfig(BaseModel):
    variables: list[dict[str, Any]] = Field(
        default_factory=list
    )  # [{name, type, default}]


class OutputConfig(BaseModel):
    mappings: dict[str, str] = Field(
        default_factory=dict
    )  # output_name -> variable_key


class SubgraphConfig(BaseModel):
    workflow_path: str
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    tool_name: str
    tool_module: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_key: str = "tool_result"


class MemoryConfig(BaseModel):
    operation: str  # "get", "put", "delete", "list", "search"
    namespace: list[str] = Field(default_factory=lambda: ["default"])
    key: str | None = None  # Variable name or literal
    value_key: str | None = None  # Variable name for value
    output_key: str = "memory_result"
    query: str | None = None  # For search
    limit: int = 10


class FanOutConfig(BaseModel):
    items_key: str  # Variable containing iterable
    target_state: str  # State to dispatch to
    item_variable: str = "item"  # Variable name in target state


class MergeConfig(BaseModel):
    reducer_key: str  # Key to reduce into
    strategy: str = "append"  # "append", "sum", "replace", "custom"


class ResourceConfig(BaseModel):
    cpu: float | None = None
    memory: float | None = None
    timeout: float | None = None
    gpu: float | None = None


class RetryConfig(BaseModel):
    max_retries: int = 3
    delay: float = 1.0
    backoff_multiplier: float = 2.0


class NodeConfig(BaseModel):
    """Union wrapper - actual config depends on node type."""

    llm: LLMConfig | None = None
    function: FunctionConfig | None = None
    conditional: ConditionalConfig | None = None
    input: InputConfig | None = None
    output: OutputConfig | None = None
    subgraph: SubgraphConfig | None = None
    tool: ToolConfig | None = None
    memory: MemoryConfig | None = None
    fan_out: FanOutConfig | None = None
    merge: MergeConfig | None = None

    def get_config(self, node_type: NodeType):
        """Get the config for a specific node type."""
        return getattr(self, node_type.value)


class Node(BaseModel):
    id: str
    type: NodeType
    position: Position = Field(default_factory=Position)
    config: NodeConfig = Field(default_factory=NodeConfig)
    resources: ResourceConfig | None = None
    retry: RetryConfig | None = None


class Edge(BaseModel):
    from_node: str
    to_node: str
    label: str | None = None  # "true"/"false" for conditionals


class WorkflowInput(BaseModel):
    name: str
    type: str = "str"
    default: Any | None = None
    description: str | None = None


class WorkflowOutput(BaseModel):
    name: str
    type: str = "str"
    description: str | None = None


class WorkflowMetadata(BaseModel):
    name: str
    description: str = ""
    author: str = ""
    tags: list[str] = Field(default_factory=list)


class StoreConfig(BaseModel):
    type: str = "memory"  # "memory", "file", etc.
    config: dict[str, Any] = Field(default_factory=dict)


class ReducerConfig(BaseModel):
    key: str
    type: str = "append"  # "append", "replace", "add", "custom"
    module: str | None = None  # For custom reducers


class AgentConfig(BaseModel):
    name: str
    class_name: str
    max_concurrent: int = 5
    store: StoreConfig | None = None
    reducers: list[ReducerConfig] = Field(default_factory=list)


class WorkflowIR(BaseModel):
    version: str = "1.0"
    metadata: WorkflowMetadata
    agent: AgentConfig
    inputs: list[WorkflowInput] = Field(default_factory=list)
    outputs: list[WorkflowOutput] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
