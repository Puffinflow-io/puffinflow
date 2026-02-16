"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/hooks/useApi";
import { useWorkflowStore } from "@/stores/workflowStore";
import { useCodegen } from "@/hooks/useCodegen";
import WorkflowCanvas from "@/components/editor/WorkflowCanvas";
import NodePalette from "@/components/editor/NodePalette";
import PropertiesPanel from "@/components/editor/PropertiesPanel";
import CodePreview from "@/components/editor/CodePreview";
import EditorToolbar from "@/components/editor/EditorToolbar";
import type { Workflow } from "@/lib/types";

export default function EditorPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [showCode, setShowCode] = useState(true);

  const store = useWorkflowStore();
  const { pythonCode: liveCode, sendYaml } = useCodegen();

  // Load workflow on mount
  useEffect(() => {
    if (!id) return;
    api.get<Workflow>(`/workflows/${id}`).then((wf) => {
      store.setWorkflowName(wf.name);
      // If workflow has YAML, try loading from IR
      // For now just set python code if available
      if (wf.generated_python) {
        store.setPythonCode(wf.generated_python);
      }
      store.setDirty(false);
    }).catch(() => {});
  }, [id]);

  // Send YAML to codegen WebSocket when store changes
  useEffect(() => {
    if (store.isDirty) {
      const ir = store.serializeToIR();
      // Convert to YAML-like JSON for the codegen service
      sendYaml(JSON.stringify(ir));
    }
  }, [store.nodes, store.edges, store.isDirty, sendYaml]);

  // Update code preview from WebSocket
  useEffect(() => {
    if (liveCode) {
      store.setPythonCode(liveCode);
    }
  }, [liveCode]);

  const handleSave = useCallback(async () => {
    if (!id) return;
    try {
      const ir = store.serializeToIR();
      await api.put(`/workflows/${id}`, {
        name: store.workflowName,
        yaml_content: JSON.stringify(ir),
      });
      store.setDirty(false);
    } catch {
      // handle error
    }
  }, [id, store]);

  const handleDeploy = useCallback(() => {
    if (id) router.push(`/deploy/${id}`);
  }, [id, router]);

  return (
    <div className="flex flex-col h-screen">
      <EditorToolbar
        onSave={handleSave}
        onDeploy={handleDeploy}
      />
      <div className="flex flex-1 overflow-hidden">
        <NodePalette />
        <div className="flex flex-col flex-1">
          <div className="flex-1 relative">
            <WorkflowCanvas />
          </div>
          {showCode && (
            <div className="h-64 border-t border-border">
              <div className="flex items-center justify-between px-3 py-1 bg-muted/50 border-b border-border">
                <span className="text-xs font-medium text-muted-foreground">Generated Python</span>
                <button
                  onClick={() => setShowCode(false)}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Hide
                </button>
              </div>
              <CodePreview />
            </div>
          )}
          {!showCode && (
            <button
              onClick={() => setShowCode(true)}
              className="px-3 py-1 text-xs text-muted-foreground hover:text-foreground border-t border-border bg-muted/50"
            >
              Show Code Preview
            </button>
          )}
        </div>
        <PropertiesPanel />
      </div>
    </div>
  );
}
