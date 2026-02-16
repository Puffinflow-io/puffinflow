"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function OutputNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const outputConfig = config?.output as Record<string, unknown> | undefined;
  const mappings = (outputConfig?.mappings as Record<string, string>) || {};
  const keys = Object.keys(mappings);

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-red-500 ring-2 ring-red-200" : "border-red-300"
      } bg-red-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-red-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-red-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">Out</span>
        </div>
        <span className="font-semibold text-sm text-red-900">Output</span>
      </div>
      {keys.length > 0 ? (
        <div className="text-xs text-red-700 space-y-0.5">
          {keys.slice(0, 3).map((key) => (
            <div key={key} className="truncate">
              {key} &larr; {mappings[key]}
            </div>
          ))}
          {keys.length > 3 && (
            <div className="text-red-500">+{keys.length - 3} more</div>
          )}
        </div>
      ) : (
        <div className="text-xs text-red-500">No mappings defined</div>
      )}
    </div>
  );
}

export const OutputNode = memo(OutputNodeComponent);
