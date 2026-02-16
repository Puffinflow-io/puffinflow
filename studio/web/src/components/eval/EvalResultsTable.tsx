"use client";
import { Badge } from "@/components/ui/Badge";

interface EvalResult {
  case_name?: string;
  case_id?: string;
  id?: string;
  scores: Record<string, number>;
  passed: boolean;
  latency_ms: number;
  actual_output?: Record<string, unknown>;
}

interface EvalResultsTableProps {
  results: EvalResult[];
}

export default function EvalResultsTable({ results }: EvalResultsTableProps) {
  if (results.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No results yet. Run the eval suite to see results.</p>
    );
  }

  const scoreKeys = results.length > 0 ? Object.keys(results[0].scores) : [];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-3 font-medium">Case</th>
            {scoreKeys.map((key) => (
              <th key={key} className="text-left py-2 px-3 font-medium">
                {key}
              </th>
            ))}
            <th className="text-left py-2 px-3 font-medium">Status</th>
            <th className="text-right py-2 px-3 font-medium">Latency</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className="border-b border-border hover:bg-muted/50">
              <td className="py-2 px-3 font-medium">{r.case_name || r.case_id || r.id || `Case ${i + 1}`}</td>
              {scoreKeys.map((key) => (
                <td key={key} className="py-2 px-3">
                  {r.scores[key]?.toFixed(2) ?? "-"}
                </td>
              ))}
              <td className="py-2 px-3">
                <Badge variant={r.passed ? "default" : "destructive"}>
                  {r.passed ? "PASS" : "FAIL"}
                </Badge>
              </td>
              <td className="py-2 px-3 text-right text-muted-foreground">
                {r.latency_ms}ms
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
