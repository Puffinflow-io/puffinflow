"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/hooks/useApi";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Label } from "@/components/ui/Label";
import { Badge } from "@/components/ui/Badge";
import LogViewer from "@/components/deploy/LogViewer";
import type { Workflow, Deployment } from "@/lib/types";

export default function DeployPage() {
  const { id } = useParams<{ id: string }>();
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [target, setTarget] = useState("docker");
  const [deploying, setDeploying] = useState(false);
  const [deployment, setDeployment] = useState<Deployment | null>(null);

  useEffect(() => {
    if (!id) return;
    api.get<Workflow>(`/workflows/${id}`).then(setWorkflow).catch(() => {});
  }, [id]);

  const handleDeploy = async () => {
    if (!id) return;
    setDeploying(true);
    try {
      const result = await api.post<Deployment>("/deploy", {
        workflow_id: id,
        target,
      });
      setDeployment(result);
    } catch {
      // handle error
    } finally {
      setDeploying(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl">
      <h1 className="text-2xl font-bold mb-2">Deploy Workflow</h1>
      <p className="text-muted-foreground mb-6">
        {workflow?.name || "Loading..."}
      </p>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Deploy Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <Label>Target Platform</Label>
            <Select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="docker">Docker (Dockerfile + FastAPI)</option>
              <option value="modal">Modal (Serverless)</option>
            </Select>
          </div>
          <Button onClick={handleDeploy} disabled={deploying}>
            {deploying ? "Deploying..." : "Deploy"}
          </Button>
        </CardContent>
      </Card>

      {deployment && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Deployment Status</CardTitle>
              <Badge variant={deployment.status === "deployed" ? "default" : "secondary"}>
                {deployment.status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {deployment.url && (
              <div>
                <Label>URL</Label>
                <p className="text-sm font-mono mt-1">{deployment.url}</p>
              </div>
            )}
            <div>
              <Label>Logs</Label>
              <LogViewer logs={deployment.logs} className="mt-1 h-48" />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
