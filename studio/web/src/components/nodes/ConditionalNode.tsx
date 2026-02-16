"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function ConditionalNodeComponent({ data, selected }: NodeProps) {
  const config = (data as Record<string, unknown>).config as Record<string, unknown> | undefined;
  const condConfig = config?.conditional as Record<string, unknown> | undefined;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-amber-500 ring-2 ring-amber-200" : "border-amber-300"
      } bg-amber-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-amber-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-amber-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">If</span>
        </div>
        <span className="font-semibold text-sm text-amber-900">Conditional</span>
      </div>
      <div className="text-xs text-amber-700 truncate max-w-[160px]">
        {condConfig?.condition
          ? (condConfig.condition as string).substring(0, 40)
          : "condition"}
      </div>
      <div className="flex justify-between mt-2 text-xs">
        <span className="text-emerald-600 font-medium">TRUE</span>
        <span className="text-red-500 font-medium">FALSE</span>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        className="!bg-emerald-500 !w-3 !h-3"
        style={{ left: "30%" }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        className="!bg-red-500 !w-3 !h-3"
        style={{ left: "70%" }}
      />
    </div>
  );
}

export const ConditionalNode = memo(ConditionalNodeComponent);
