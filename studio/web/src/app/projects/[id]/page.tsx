"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api } from "@/hooks/useApi";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Label } from "@/components/ui/Label";
import { Badge } from "@/components/ui/Badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import type { Project, Workflow, EvalSuite, Deployment } from "@/lib/types";

export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [suites, setSuites] = useState<EvalSuite[]>([]);
  const [showNewWorkflow, setShowNewWorkflow] = useState(false);
  const [newWorkflowName, setNewWorkflowName] = useState("");
  const [activeTab, setActiveTab] = useState("workflows");

  useEffect(() => {
    if (!id) return;
    api.get<Project>(`/projects/${id}`).then(setProject).catch(() => {});
    api.get<Workflow[]>(`/workflows?project_id=${id}`).then(setWorkflows).catch(() => {});
    api.get<EvalSuite[]>(`/eval/suites?project_id=${id}`).then(setSuites).catch(() => {});
  }, [id]);

  const createWorkflow = async () => {
    if (!newWorkflowName.trim() || !id) return;
    try {
      const wf = await api.post<Workflow>("/workflows", {
        project_id: id,
        name: newWorkflowName,
      });
      setWorkflows((prev) => [wf, ...prev]);
      setNewWorkflowName("");
      setShowNewWorkflow(false);
    } catch {
      // handle
    }
  };

  const deleteWorkflow = async (wfId: string) => {
    try {
      await api.delete(`/workflows/${wfId}`);
      setWorkflows((prev) => prev.filter((w) => w.id !== wfId));
    } catch {
      // handle
    }
  };

  if (!project) {
    return <div className="p-8 text-muted-foreground">Loading...</div>;
  }

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">{project.name}</h1>
        <p className="text-muted-foreground mt-1">{project.description || "No description"}</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="evals">Eval Suites</TabsTrigger>
        </TabsList>

        <TabsContent value="workflows" className="mt-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Workflows</h2>
            <Button size="sm" onClick={() => setShowNewWorkflow(!showNewWorkflow)}>
              + New Workflow
            </Button>
          </div>

          {showNewWorkflow && (
            <Card className="mb-4">
              <CardContent className="pt-4 flex items-end gap-3">
                <div className="flex-1 space-y-1">
                  <Label>Workflow Name</Label>
                  <Input
                    value={newWorkflowName}
                    onChange={(e) => setNewWorkflowName(e.target.value)}
                    placeholder="my-workflow"
                  />
                </div>
                <Button onClick={createWorkflow}>Create</Button>
                <Button variant="outline" onClick={() => setShowNewWorkflow(false)}>Cancel</Button>
              </CardContent>
            </Card>
          )}

          <div className="space-y-2">
            {workflows.map((wf) => (
              <Card key={wf.id}>
                <CardContent className="py-3 flex items-center justify-between">
                  <div>
                    <Link href={`/editor/${wf.id}`} className="font-medium hover:underline">
                      {wf.name}
                    </Link>
                    <span className="text-xs text-muted-foreground ml-2">v{wf.version}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Link href={`/editor/${wf.id}`}>
                      <Button variant="outline" size="sm">Edit</Button>
                    </Link>
                    <Link href={`/deploy/${wf.id}`}>
                      <Button variant="outline" size="sm">Deploy</Button>
                    </Link>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive"
                      onClick={() => deleteWorkflow(wf.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
            {workflows.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-8">
                No workflows yet. Create one to get started.
              </p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="evals" className="mt-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Eval Suites</h2>
            <Button size="sm">+ New Suite</Button>
          </div>
          <div className="space-y-2">
            {suites.map((suite) => (
              <Card key={suite.id}>
                <CardContent className="py-3 flex items-center justify-between">
                  <Link href={`/eval/${suite.id}`} className="font-medium hover:underline">
                    {suite.name}
                  </Link>
                  <Link href={`/eval/${suite.id}`}>
                    <Button variant="outline" size="sm">View</Button>
                  </Link>
                </CardContent>
              </Card>
            ))}
            {suites.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-8">
                No eval suites yet.
              </p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
