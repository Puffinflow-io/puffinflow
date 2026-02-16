"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function FunctionNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const fnConfig = config?.function as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-blue-500 ring-2 ring-blue-200" : "border-blue-300"
      } bg-blue-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-blue-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-blue-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">Fn</span>
        </div>
        <span className="font-semibold text-sm text-blue-900">Function</span>
      </div>
      <div className="text-xs text-blue-700 truncate">
        {fnConfig?.module ? (fnConfig.module as string) : "Custom Code"}
      </div>
      {fnConfig?.output_key && (
        <div className="text-xs text-blue-500 mt-1 truncate">
          output: {fnConfig.output_key as string}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-blue-500 !w-3 !h-3" />
    </div>
  );
}

export const FunctionNode = memo(FunctionNodeComponent);
