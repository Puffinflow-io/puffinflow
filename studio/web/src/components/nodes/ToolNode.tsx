"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function ToolNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const toolConfig = config?.tool as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-pink-500 ring-2 ring-pink-200" : "border-pink-300"
      } bg-pink-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-pink-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-pink-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">T</span>
        </div>
        <span className="font-semibold text-sm text-pink-900">Tool</span>
      </div>
      <div className="text-xs text-pink-700 truncate">
        {toolConfig?.tool_name
          ? (toolConfig.tool_name as string)
          : "No tool selected"}
      </div>
      {toolConfig?.output_key && (
        <div className="text-xs text-pink-500 mt-1 truncate">
          output: {toolConfig.output_key as string}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-pink-500 !w-3 !h-3" />
    </div>
  );
}

export const ToolNode = memo(ToolNodeComponent);
