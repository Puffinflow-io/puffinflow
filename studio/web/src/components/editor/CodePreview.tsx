"use client";
import { useWorkflowStore } from "@/stores/workflowStore";

export default function CodePreview() {
  const pythonCode = useWorkflowStore((s) => s.pythonCode);

  const code = pythonCode || "# No code generated yet.\n# Add nodes and edges to see generated Python code.";

  return (
    <div className="h-full overflow-auto p-4 bg-card">
      <pre className="text-xs font-mono whitespace-pre-wrap text-foreground">
        <code>{code}</code>
      </pre>
    </div>
  );
}
