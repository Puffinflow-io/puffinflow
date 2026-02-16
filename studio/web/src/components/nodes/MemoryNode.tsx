"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function MemoryNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const memConfig = config?.memory as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-teal-500 ring-2 ring-teal-200" : "border-teal-300"
      } bg-teal-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-teal-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-teal-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">M</span>
        </div>
        <span className="font-semibold text-sm text-teal-900">Memory</span>
      </div>
      <div className="text-xs text-teal-700 truncate">
        {memConfig?.operation
          ? (memConfig.operation as string).toUpperCase()
          : "GET"}
      </div>
      {memConfig?.namespace && (
        <div className="text-xs text-teal-500 mt-1 truncate">
          ns: {(memConfig.namespace as string[]).join("/")}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-teal-500 !w-3 !h-3" />
    </div>
  );
}

export const MemoryNode = memo(MemoryNodeComponent);
