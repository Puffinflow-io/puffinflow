"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function FanOutNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const fanOutConfig = config?.fan_out as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-orange-500 ring-2 ring-orange-200" : "border-orange-300"
      } bg-orange-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-orange-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-orange-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">FO</span>
        </div>
        <span className="font-semibold text-sm text-orange-900">Fan Out</span>
      </div>
      <div className="text-xs text-orange-700 truncate">
        items: {fanOutConfig?.items_key ? (fanOutConfig.items_key as string) : "items"}
      </div>
      {fanOutConfig?.target_state && (
        <div className="text-xs text-orange-500 mt-1 truncate">
          target: {fanOutConfig.target_state as string}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-orange-500 !w-3 !h-3" />
    </div>
  );
}

export const FanOutNode = memo(FanOutNodeComponent);
