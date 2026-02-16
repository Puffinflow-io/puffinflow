"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function MergeNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const mergeConfig = config?.merge as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-cyan-500 ring-2 ring-cyan-200" : "border-cyan-300"
      } bg-cyan-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-cyan-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-cyan-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">MG</span>
        </div>
        <span className="font-semibold text-sm text-cyan-900">Merge</span>
      </div>
      <div className="text-xs text-cyan-700 truncate">
        strategy: {mergeConfig?.strategy ? (mergeConfig.strategy as string) : "append"}
      </div>
      {mergeConfig?.reducer_key && (
        <div className="text-xs text-cyan-500 mt-1 truncate">
          key: {mergeConfig.reducer_key as string}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-cyan-500 !w-3 !h-3" />
    </div>
  );
}

export const MergeNode = memo(MergeNodeComponent);
