"use client";
import { useWorkflowStore } from "@/stores/workflowStore";
import { NodeType, type NodeConfig } from "@/lib/types";
import { getNodeTypeInfo } from "@/lib/nodeRegistry";
import { Input } from "@/components/ui/Input";
import { Textarea } from "@/components/ui/Textarea";
import { Label } from "@/components/ui/Label";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";

function LLMProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const llm = config.llm || { model: "gpt-4", user_prompt: "", temperature: 0.7, output_key: "response" };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, llm: { ...llm, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="model">Model</Label>
        <Select id="model" value={llm.model} onChange={(e) => update({ model: e.target.value })}>
          <option value="gpt-4">gpt-4</option>
          <option value="gpt-4o">gpt-4o</option>
          <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
          <option value="claude-3-opus">claude-3-opus</option>
          <option value="claude-3-sonnet">claude-3-sonnet</option>
        </Select>
      </div>
      <div>
        <Label htmlFor="system_prompt">System Prompt</Label>
        <Textarea
          id="system_prompt"
          value={llm.system_prompt || ""}
          onChange={(e) => update({ system_prompt: e.target.value })}
          placeholder="Optional system prompt..."
          rows={3}
        />
      </div>
      <div>
        <Label htmlFor="user_prompt">User Prompt</Label>
        <Textarea
          id="user_prompt"
          value={llm.user_prompt}
          onChange={(e) => update({ user_prompt: e.target.value })}
          placeholder="Enter your prompt template..."
          rows={4}
        />
      </div>
      <div>
        <Label htmlFor="temperature">Temperature: {llm.temperature}</Label>
        <input
          id="temperature"
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={llm.temperature}
          onChange={(e) => update({ temperature: parseFloat(e.target.value) })}
          className="w-full"
        />
      </div>
      <div>
        <Label htmlFor="output_key">Output Key</Label>
        <Input
          id="output_key"
          value={llm.output_key}
          onChange={(e) => update({ output_key: e.target.value })}
        />
      </div>
    </div>
  );
}

function FunctionProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const fn = config.function || { code: "", input_keys: [], output_key: "result" };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, function: { ...fn, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="code">Code</Label>
        <Textarea
          id="code"
          value={fn.code || ""}
          onChange={(e) => update({ code: e.target.value })}
          placeholder="# Your Python code here"
          rows={8}
          className="font-mono text-xs"
        />
      </div>
      <div>
        <Label htmlFor="input_keys">Input Keys (comma-separated)</Label>
        <Input
          id="input_keys"
          value={fn.input_keys.join(", ")}
          onChange={(e) =>
            update({ input_keys: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })
          }
        />
      </div>
      <div>
        <Label htmlFor="fn_output_key">Output Key</Label>
        <Input
          id="fn_output_key"
          value={fn.output_key || ""}
          onChange={(e) => update({ output_key: e.target.value })}
        />
      </div>
    </div>
  );
}

function ConditionalProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const cond = config.conditional || { condition: "", true_target: "", false_target: "" };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, conditional: { ...cond, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="condition">Condition</Label>
        <Textarea
          id="condition"
          value={cond.condition}
          onChange={(e) => update({ condition: e.target.value })}
          placeholder="ctx.get_variable('value') > 0"
          rows={3}
          className="font-mono text-xs"
        />
      </div>
      <div>
        <Label htmlFor="true_target">True Target</Label>
        <Input
          id="true_target"
          value={cond.true_target}
          onChange={(e) => update({ true_target: e.target.value })}
          placeholder="Node ID for true branch"
        />
      </div>
      <div>
        <Label htmlFor="false_target">False Target</Label>
        <Input
          id="false_target"
          value={cond.false_target}
          onChange={(e) => update({ false_target: e.target.value })}
          placeholder="Node ID for false branch"
        />
      </div>
    </div>
  );
}

function ToolProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const tool = config.tool || { tool_name: "", parameters: {}, output_key: "tool_result" };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, tool: { ...tool, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="tool_name">Tool Name</Label>
        <Input
          id="tool_name"
          value={tool.tool_name}
          onChange={(e) => update({ tool_name: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="tool_module">Tool Module</Label>
        <Input
          id="tool_module"
          value={tool.tool_module || ""}
          onChange={(e) => update({ tool_module: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="tool_params">Parameters (JSON)</Label>
        <Textarea
          id="tool_params"
          value={JSON.stringify(tool.parameters, null, 2)}
          onChange={(e) => {
            try {
              update({ parameters: JSON.parse(e.target.value) });
            } catch {
              // Invalid JSON, ignore
            }
          }}
          rows={4}
          className="font-mono text-xs"
        />
      </div>
      <div>
        <Label htmlFor="tool_output_key">Output Key</Label>
        <Input
          id="tool_output_key"
          value={tool.output_key}
          onChange={(e) => update({ output_key: e.target.value })}
        />
      </div>
    </div>
  );
}

function MemoryProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const mem = config.memory || { operation: "get" as const, namespace: ["default"], output_key: "memory_result", limit: 10 };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, memory: { ...mem, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="operation">Operation</Label>
        <Select
          id="operation"
          value={mem.operation}
          onChange={(e) => update({ operation: e.target.value })}
        >
          <option value="get">get</option>
          <option value="put">put</option>
          <option value="delete">delete</option>
          <option value="list">list</option>
          <option value="search">search</option>
        </Select>
      </div>
      <div>
        <Label htmlFor="namespace">Namespace (comma-separated)</Label>
        <Input
          id="namespace"
          value={mem.namespace.join(", ")}
          onChange={(e) =>
            update({ namespace: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })
          }
        />
      </div>
      <div>
        <Label htmlFor="mem_output_key">Output Key</Label>
        <Input
          id="mem_output_key"
          value={mem.output_key}
          onChange={(e) => update({ output_key: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="limit">Limit</Label>
        <Input
          id="limit"
          type="number"
          value={mem.limit}
          onChange={(e) => update({ limit: parseInt(e.target.value) || 10 })}
        />
      </div>
    </div>
  );
}

function SubgraphProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const sub = config.subgraph || { workflow_path: "", input_mapping: {}, output_mapping: {} };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, subgraph: { ...sub, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="workflow_path">Workflow Path</Label>
        <Input
          id="workflow_path"
          value={sub.workflow_path}
          onChange={(e) => update({ workflow_path: e.target.value })}
          placeholder="path/to/workflow.yaml"
        />
      </div>
      <div>
        <Label htmlFor="input_mapping">Input Mapping (JSON)</Label>
        <Textarea
          id="input_mapping"
          value={JSON.stringify(sub.input_mapping, null, 2)}
          onChange={(e) => {
            try {
              update({ input_mapping: JSON.parse(e.target.value) });
            } catch {
              // Invalid JSON
            }
          }}
          rows={3}
          className="font-mono text-xs"
        />
      </div>
      <div>
        <Label htmlFor="output_mapping">Output Mapping (JSON)</Label>
        <Textarea
          id="output_mapping"
          value={JSON.stringify(sub.output_mapping, null, 2)}
          onChange={(e) => {
            try {
              update({ output_mapping: JSON.parse(e.target.value) });
            } catch {
              // Invalid JSON
            }
          }}
          rows={3}
          className="font-mono text-xs"
        />
      </div>
    </div>
  );
}

function FanOutProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const fanOut = config.fan_out || { items_key: "items", target_state: "", item_variable: "item" };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, fan_out: { ...fanOut, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="items_key">Items Key</Label>
        <Input
          id="items_key"
          value={fanOut.items_key}
          onChange={(e) => update({ items_key: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="target_state">Target State</Label>
        <Input
          id="target_state"
          value={fanOut.target_state}
          onChange={(e) => update({ target_state: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="item_variable">Item Variable</Label>
        <Input
          id="item_variable"
          value={fanOut.item_variable}
          onChange={(e) => update({ item_variable: e.target.value })}
        />
      </div>
    </div>
  );
}

function MergeProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const merge = config.merge || { reducer_key: "results", strategy: "append" };
  const update = (patch: Record<string, unknown>) => {
    onChange({ ...config, merge: { ...merge, ...patch } });
  };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="reducer_key">Reducer Key</Label>
        <Input
          id="reducer_key"
          value={merge.reducer_key}
          onChange={(e) => update({ reducer_key: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="strategy">Strategy</Label>
        <Select
          id="strategy"
          value={merge.strategy}
          onChange={(e) => update({ strategy: e.target.value })}
        >
          <option value="append">append</option>
          <option value="merge">merge</option>
          <option value="replace">replace</option>
        </Select>
      </div>
    </div>
  );
}

function InputProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const input = config.input || { variables: [] };
  const update = (variables: Array<{ name: string; type: string; default?: unknown }>) => {
    onChange({ ...config, input: { variables } });
  };

  const addVariable = () => {
    update([...input.variables, { name: "", type: "string" }]);
  };

  const removeVariable = (index: number) => {
    update(input.variables.filter((_, i) => i !== index));
  };

  const updateVariable = (index: number, patch: Record<string, unknown>) => {
    update(input.variables.map((v, i) => (i === index ? { ...v, ...patch } : v)));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Label>Variables</Label>
        <Button variant="outline" size="sm" onClick={addVariable}>
          + Add
        </Button>
      </div>
      {input.variables.map((v, i) => (
        <div key={i} className="flex gap-2 items-end">
          <div className="flex-1">
            <Input
              placeholder="Name"
              value={v.name}
              onChange={(e) => updateVariable(i, { name: e.target.value })}
            />
          </div>
          <div className="w-24">
            <Select value={v.type} onChange={(e) => updateVariable(i, { type: e.target.value })}>
              <option value="string">string</option>
              <option value="int">int</option>
              <option value="float">float</option>
              <option value="bool">bool</option>
              <option value="list">list</option>
              <option value="dict">dict</option>
            </Select>
          </div>
          <Button variant="ghost" size="sm" onClick={() => removeVariable(i)}>
            X
          </Button>
        </div>
      ))}
    </div>
  );
}

function OutputProperties({ config, onChange }: { config: NodeConfig; onChange: (c: NodeConfig) => void }) {
  const output = config.output || { mappings: {} };

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="mappings">Output Mappings (JSON)</Label>
        <Textarea
          id="mappings"
          value={JSON.stringify(output.mappings, null, 2)}
          onChange={(e) => {
            try {
              onChange({ ...config, output: { mappings: JSON.parse(e.target.value) } });
            } catch {
              // Invalid JSON
            }
          }}
          rows={6}
          className="font-mono text-xs"
          placeholder='{\n  "result": "response"\n}'
        />
      </div>
    </div>
  );
}

const propertyRenderers: Record<
  NodeType,
  (props: { config: NodeConfig; onChange: (c: NodeConfig) => void }) => JSX.Element
> = {
  [NodeType.LLM]: LLMProperties,
  [NodeType.FUNCTION]: FunctionProperties,
  [NodeType.CONDITIONAL]: ConditionalProperties,
  [NodeType.INPUT]: InputProperties,
  [NodeType.OUTPUT]: OutputProperties,
  [NodeType.SUBGRAPH]: SubgraphProperties,
  [NodeType.TOOL]: ToolProperties,
  [NodeType.MEMORY]: MemoryProperties,
  [NodeType.FAN_OUT]: FanOutProperties,
  [NodeType.MERGE]: MergeProperties,
};

export default function PropertiesPanel() {
  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const nodes = useWorkflowStore((s) => s.nodes);
  const updateNodeConfig = useWorkflowStore((s) => s.updateNodeConfig);
  const removeNode = useWorkflowStore((s) => s.removeNode);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  if (!selectedNode) {
    return (
      <div className="w-72 border-l border-border bg-card p-4 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Select a node to edit its properties</p>
      </div>
    );
  }

  const info = getNodeTypeInfo(selectedNode.type);
  const Renderer = propertyRenderers[selectedNode.type];

  return (
    <div className="w-72 border-l border-border bg-card overflow-y-auto">
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 mb-1">
          <div
            className="w-6 h-6 rounded flex items-center justify-center text-white text-xs font-bold"
            style={{ backgroundColor: info.color }}
          >
            {info.label.substring(0, 2).toUpperCase()}
          </div>
          <span className="font-semibold text-sm">{info.label}</span>
        </div>
        <p className="text-xs text-muted-foreground">{selectedNode.id}</p>
      </div>
      <div className="p-4">
        <Renderer config={selectedNode.config} onChange={(c) => updateNodeConfig(selectedNode.id, c)} />
      </div>
      <div className="p-4 border-t border-border">
        <Button
          variant="destructive"
          size="sm"
          className="w-full"
          onClick={() => removeNode(selectedNode.id)}
        >
          Delete Node
        </Button>
      </div>
    </div>
  );
}
