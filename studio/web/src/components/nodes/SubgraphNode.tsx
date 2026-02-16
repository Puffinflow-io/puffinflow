"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function SubgraphNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const subConfig = config?.subgraph as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-indigo-500 ring-2 ring-indigo-200" : "border-indigo-300"
      } bg-indigo-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-indigo-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-indigo-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">SG</span>
        </div>
        <span className="font-semibold text-sm text-indigo-900">Subgraph</span>
      </div>
      <div className="text-xs text-indigo-700 truncate">
        {subConfig?.workflow_path
          ? (subConfig.workflow_path as string)
          : "No workflow selected"}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-indigo-500 !w-3 !h-3" />
    </div>
  );
}

export const SubgraphNode = memo(SubgraphNodeComponent);
