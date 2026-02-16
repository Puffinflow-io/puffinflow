"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/hooks/useApi";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Textarea } from "@/components/ui/Textarea";
import { Label } from "@/components/ui/Label";
import type { Project } from "@/lib/types";

export default function DashboardPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  useEffect(() => {
    api.get<Project[]>("/projects").then(setProjects).catch(() => {});
  }, []);

  const createProject = async () => {
    if (!newName.trim()) return;
    try {
      const project = await api.post<Project>("/projects", {
        name: newName,
        description: newDesc,
      });
      setProjects((prev) => [project, ...prev]);
      setNewName("");
      setNewDesc("");
      setShowNew(false);
    } catch {
      // handle error
    }
  };

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Projects</h1>
          <p className="text-muted-foreground mt-1">Manage your AI agent workflows</p>
        </div>
        <Button onClick={() => setShowNew(!showNew)}>
          + New Project
        </Button>
      </div>

      {showNew && (
        <Card className="mb-6">
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-1">
              <Label>Project Name</Label>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="my-agent-project"
              />
            </div>
            <div className="space-y-1">
              <Label>Description</Label>
              <Textarea
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="A brief description..."
                rows={2}
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={createProject}>Create</Button>
              <Button variant="outline" onClick={() => setShowNew(false)}>Cancel</Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects.map((project) => (
          <Link key={project.id} href={`/projects/${project.id}`}>
            <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
              <CardHeader>
                <CardTitle className="text-lg">{project.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {project.description || "No description"}
                </p>
                <p className="text-xs text-muted-foreground mt-3">
                  Updated {new Date(project.updated_at).toLocaleDateString()}
                </p>
              </CardContent>
            </Card>
          </Link>
        ))}
        {projects.length === 0 && !showNew && (
          <div className="col-span-full text-center py-12 text-muted-foreground">
            <p className="text-lg mb-2">No projects yet</p>
            <p className="text-sm">Create your first project to get started</p>
          </div>
        )}
      </div>
    </div>
  );
}
