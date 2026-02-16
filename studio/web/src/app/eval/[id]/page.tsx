"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/hooks/useApi";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Input } from "@/components/ui/Input";
import { Label } from "@/components/ui/Label";
import EvalSuiteEditor from "@/components/eval/EvalSuiteEditor";
import EvalResultsTable from "@/components/eval/EvalResultsTable";
import type { EvalSuite, EvalRun, Workflow } from "@/lib/types";

interface EvalCase {
  id: string;
  suite_id: string;
  name: string;
  input_data: Record<string, unknown>;
  expected_output: Record<string, unknown>;
  tags: string[];
}

interface EvalRunWithResults extends EvalRun {
  results: Array<{
    id: string;
    case_id: string;
    actual_output: Record<string, unknown>;
    scores: Record<string, number>;
    latency_ms: number;
    passed: boolean;
  }>;
}

export default function EvalPage() {
  const { id } = useParams<{ id: string }>();
  const [suite, setSuite] = useState<EvalSuite | null>(null);
  const [cases, setCases] = useState<EvalCase[]>([]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<EvalRunWithResults | null>(null);
  const [workflowId, setWorkflowId] = useState("");
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (!id) return;
    api.get<EvalSuite>(`/eval/suites/${id}`).then(setSuite).catch(() => {});
    api.get<EvalCase[]>(`/eval/suites/${id}/cases`).then(setCases).catch(() => {});
  }, [id]);

  const runSuite = async () => {
    if (!id || !workflowId) return;
    setRunning(true);
    try {
      const run = await api.post<EvalRunWithResults>(`/eval/suites/${id}/run`, {
        workflow_id: workflowId,
        parallel: 1,
      });
      setSelectedRun(run);
    } catch {
      // handle error
    } finally {
      setRunning(false);
    }
  };

  const viewRun = async (runId: string) => {
    try {
      const run = await api.get<EvalRunWithResults>(`/eval/runs/${runId}`);
      setSelectedRun(run);
    } catch {
      // handle error
    }
  };

  return (
    <div className="p-8 max-w-4xl">
      <h1 className="text-2xl font-bold mb-2">Eval Suite</h1>
      <p className="text-muted-foreground mb-6">{suite?.name || "Loading..."}</p>

      {suite && (
        <EvalSuiteEditor
          suite={suite}
          onSave={async (data) => {
            try {
              const updated = await api.put<EvalSuite>(`/eval/suites/${id}`, data);
              setSuite(updated);
            } catch {
              // handle
            }
          }}
        />
      )}

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Test Cases ({cases.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {cases.map((c) => (
              <div key={c.id} className="flex items-center justify-between py-2 px-3 bg-muted/50 rounded">
                <div>
                  <span className="text-sm font-medium">{c.name}</span>
                  {c.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="ml-2 text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                <span className="text-xs text-muted-foreground">
                  {Object.keys(c.input_data).length} inputs
                </span>
              </div>
            ))}
            {cases.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-4">
                No test cases yet.
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Run Evaluation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <Label>Workflow ID</Label>
            <Input
              value={workflowId}
              onChange={(e) => setWorkflowId(e.target.value)}
              placeholder="Enter workflow ID to test against"
            />
          </div>
          <Button onClick={runSuite} disabled={running || !workflowId}>
            {running ? "Running..." : "Run Suite"}
          </Button>
        </CardContent>
      </Card>

      {selectedRun && (
        <Card className="mt-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Results</CardTitle>
              <Badge variant={selectedRun.status === "completed" ? "default" : "destructive"}>
                {selectedRun.status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {selectedRun.summary_scores && (
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-2xl font-bold">{selectedRun.summary_scores.total ?? 0}</div>
                  <div className="text-xs text-muted-foreground">Total</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{selectedRun.summary_scores.passed ?? 0}</div>
                  <div className="text-xs text-muted-foreground">Passed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-600">{selectedRun.summary_scores.failed ?? 0}</div>
                  <div className="text-xs text-muted-foreground">Failed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{((selectedRun.summary_scores.avg_score ?? 0) * 100).toFixed(0)}%</div>
                  <div className="text-xs text-muted-foreground">Avg Score</div>
                </div>
              </div>
            )}
            {selectedRun.results && selectedRun.results.length > 0 && (
              <EvalResultsTable results={selectedRun.results} />
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
