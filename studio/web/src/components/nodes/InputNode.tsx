"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function InputNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const inputConfig = config?.input as Record<string, unknown> | undefined;
  const variables = (inputConfig?.variables as Array<Record<string, unknown>>) || [];

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-emerald-500 ring-2 ring-emerald-200" : "border-emerald-300"
      } bg-emerald-50`}
    >
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-emerald-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">In</span>
        </div>
        <span className="font-semibold text-sm text-emerald-900">Input</span>
      </div>
      {variables.length > 0 ? (
        <div className="text-xs text-emerald-700 space-y-0.5">
          {variables.slice(0, 3).map((v, i) => (
            <div key={i} className="truncate">
              {v.name as string}: {v.type as string}
            </div>
          ))}
          {variables.length > 3 && (
            <div className="text-emerald-500">+{variables.length - 3} more</div>
          )}
        </div>
      ) : (
        <div className="text-xs text-emerald-500">No variables defined</div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-emerald-500 !w-3 !h-3" />
    </div>
  );
}

export const InputNode = memo(InputNodeComponent);
