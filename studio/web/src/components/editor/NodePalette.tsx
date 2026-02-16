"use client";
import { type DragEvent } from "react";
import { getAllNodeTypes, type NodeTypeInfo } from "@/lib/nodeRegistry";

function PaletteItem({ info }: { info: NodeTypeInfo }) {
  const onDragStart = (event: DragEvent) => {
    event.dataTransfer.setData("application/reactflow", info.type);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className="flex items-center gap-3 p-3 rounded-lg border border-border bg-card hover:bg-accent cursor-grab active:cursor-grabbing transition-colors"
    >
      <div
        className="w-8 h-8 rounded flex items-center justify-center text-white text-xs font-bold shrink-0"
        style={{ backgroundColor: info.color }}
      >
        {info.label.substring(0, 2).toUpperCase()}
      </div>
      <div className="min-w-0">
        <div className="text-sm font-medium truncate">{info.label}</div>
        <div className="text-xs text-muted-foreground truncate">{info.description}</div>
      </div>
    </div>
  );
}

export default function NodePalette() {
  const nodeTypes = getAllNodeTypes();

  return (
    <div className="w-64 border-r border-border bg-card overflow-y-auto p-4 space-y-2">
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
        Node Types
      </h2>
      {nodeTypes.map((info) => (
        <PaletteItem key={info.type} info={info} />
      ))}
    </div>
  );
}
