"use client";
import { useCallback } from "react";
import { useWorkflowStore } from "@/stores/workflowStore";
import { api } from "@/hooks/useApi";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

interface EditorToolbarProps {
  onSave?: () => void;
  onDeploy?: () => void;
}

export default function EditorToolbar({ onSave, onDeploy }: EditorToolbarProps) {
  const workflowName = useWorkflowStore((s) => s.workflowName);
  const setWorkflowName = useWorkflowStore((s) => s.setWorkflowName);
  const isDirty = useWorkflowStore((s) => s.isDirty);
  const serializeToIR = useWorkflowStore((s) => s.serializeToIR);
  const setDirty = useWorkflowStore((s) => s.setDirty);
  const setPythonCode = useWorkflowStore((s) => s.setPythonCode);
  const workflowId = useWorkflowStore((s) => s.workflowId);
  const projectId = useWorkflowStore((s) => s.projectId);

  const handleSave = useCallback(async () => {
    if (onSave) {
      onSave();
      return;
    }
    const ir = serializeToIR();
    try {
      if (workflowId && projectId) {
        await api.put(`/projects/${projectId}/workflows/${workflowId}`, {
          yaml_content: JSON.stringify(ir),
        });
      }
      setDirty(false);
    } catch (err) {
      console.error("Failed to save workflow:", err);
    }
  }, [onSave, serializeToIR, workflowId, projectId, setDirty]);

  const handleGenerate = useCallback(async () => {
    const ir = serializeToIR();
    try {
      const result = await api.post<{ python: string }>("/codegen/generate", {
        yaml_content: JSON.stringify(ir),
      });
      setPythonCode(result.python);
    } catch (err) {
      console.error("Failed to generate code:", err);
    }
  }, [serializeToIR, setPythonCode]);

  const handleDeploy = useCallback(() => {
    if (onDeploy) {
      onDeploy();
    }
  }, [onDeploy]);

  return (
    <div className="h-14 border-b border-border bg-card flex items-center px-4 gap-3">
      <Input
        value={workflowName}
        onChange={(e) => setWorkflowName(e.target.value)}
        className="w-64 h-8 text-sm font-medium"
      />
      {isDirty && (
        <span className="text-xs text-amber-500 font-medium">Unsaved changes</span>
      )}
      <div className="flex-1" />
      <Button variant="outline" size="sm" onClick={handleGenerate}>
        Generate
      </Button>
      <Button variant="outline" size="sm" onClick={handleSave}>
        Save
      </Button>
      <Button size="sm" onClick={handleDeploy}>
        Deploy
      </Button>
    </div>
  );
}
