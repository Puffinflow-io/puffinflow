"use client";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function LLMNodeComponent({ data, selected }: NodeProps) {
  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-sm min-w-[180px] ${
        selected ? "border-violet-500 ring-2 ring-violet-200" : "border-violet-300"
      } bg-violet-50`}
    >
      <Handle type="target" position={Position.Top} className="!bg-violet-500 !w-3 !h-3" />
      <div className="flex items-center gap-2 mb-1">
        <div className="w-6 h-6 rounded bg-violet-500 flex items-center justify-center">
          <span className="text-white text-xs font-bold">AI</span>
        </div>
        <span className="font-semibold text-sm text-violet-900">LLM</span>
      </div>
      <div className="text-xs text-violet-700 truncate">
        {(data as Record<string, unknown>).config
          ? ((data as Record<string, unknown>).config as Record<string, unknown>).llm
            ? (((data as Record<string, unknown>).config as Record<string, unknown>).llm as Record<string, unknown>).model as string
            : "gpt-4"
          : "gpt-4"}
      </div>
      {(data as Record<string, unknown>).config &&
        ((data as Record<string, unknown>).config as Record<string, unknown>).llm &&
        (((data as Record<string, unknown>).config as Record<string, unknown>).llm as Record<string, unknown>).user_prompt && (
          <div className="text-xs text-violet-500 mt-1 truncate max-w-[160px]">
            {(
              (((data as Record<string, unknown>).config as Record<string, unknown>).llm as Record<string, unknown>)
                .user_prompt as string
            ).substring(0, 40)}
            ...
          </div>
        )}
      <Handle type="source" position={Position.Bottom} className="!bg-violet-500 !w-3 !h-3" />
    </div>
  );
}

export const LLMNode = memo(LLMNodeComponent);
