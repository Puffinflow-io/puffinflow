"use client";
import { useState } from "react";
import { api } from "@/hooks/useApi";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import { Badge } from "@/components/ui/Badge";
import type { Deployment } from "@/lib/types";

interface DeployPanelProps {
  workflowId: string;
  deployments: Deployment[];
  onDeployStarted: (deployment: Deployment) => void;
}

export default function DeployPanel({ workflowId, deployments, onDeployStarted }: DeployPanelProps) {
  const [target, setTarget] = useState("modal");
  const [isDeploying, setIsDeploying] = useState(false);

  const handleDeploy = async () => {
    setIsDeploying(true);
    try {
      const deployment = await api.post<Deployment>("/deploy", {
        workflow_id: workflowId,
        target,
      });
      onDeployStarted(deployment);
    } catch (err) {
      console.error("Deploy failed:", err);
    } finally {
      setIsDeploying(false);
    }
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "running":
        return "default" as const;
      case "completed":
        return "secondary" as const;
      case "failed":
        return "destructive" as const;
      default:
        return "outline" as const;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-end gap-4">
        <Select
          label="Deploy Target"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="w-48"
        >
          <option value="modal">Modal</option>
          <option value="docker">Docker</option>
        </Select>
        <Button onClick={handleDeploy} disabled={isDeploying}>
          {isDeploying ? "Deploying..." : "Deploy"}
        </Button>
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold">Deployment History</h3>
        {deployments.length === 0 ? (
          <p className="text-sm text-muted-foreground">No deployments yet</p>
        ) : (
          <div className="space-y-2">
            {deployments.map((d) => (
              <div key={d.id} className="flex items-center justify-between p-3 rounded-lg border border-border">
                <div>
                  <div className="text-sm font-medium">{d.target}</div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(d.created_at).toLocaleString()}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {d.url && (
                    <a
                      href={d.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-primary hover:underline"
                    >
                      URL
                    </a>
                  )}
                  <Badge variant={statusColor(d.status)}>{d.status}</Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
