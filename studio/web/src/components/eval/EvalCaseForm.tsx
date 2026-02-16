"use client";
import { Input } from "@/components/ui/Input";
import { Textarea } from "@/components/ui/Textarea";
import { Label } from "@/components/ui/Label";
import { Button } from "@/components/ui/Button";

interface EvalCase {
  name: string;
  input: string;
  expected_output: string;
  tags: string[];
}

interface EvalCaseFormProps {
  evalCase: EvalCase;
  onChange: (updated: EvalCase) => void;
  onRemove: () => void;
}

export default function EvalCaseForm({ evalCase, onChange, onRemove }: EvalCaseFormProps) {
  return (
    <div className="p-4 rounded-lg border border-border space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex-1 mr-4">
          <Label htmlFor="case-name">Case Name</Label>
          <Input
            id="case-name"
            value={evalCase.name}
            onChange={(e) => onChange({ ...evalCase, name: e.target.value })}
            placeholder="Test case name"
          />
        </div>
        <Button variant="ghost" size="sm" onClick={onRemove}>
          Remove
        </Button>
      </div>

      <div>
        <Label htmlFor="case-input">Input (JSON)</Label>
        <Textarea
          id="case-input"
          value={evalCase.input}
          onChange={(e) => onChange({ ...evalCase, input: e.target.value })}
          rows={3}
          className="font-mono text-xs"
          placeholder='{"query": "Hello"}'
        />
      </div>

      <div>
        <Label htmlFor="case-expected">Expected Output (JSON)</Label>
        <Textarea
          id="case-expected"
          value={evalCase.expected_output}
          onChange={(e) => onChange({ ...evalCase, expected_output: e.target.value })}
          rows={3}
          className="font-mono text-xs"
          placeholder='{"response": "Hello, world!"}'
        />
      </div>

      <div>
        <Label htmlFor="case-tags">Tags (comma-separated)</Label>
        <Input
          id="case-tags"
          value={evalCase.tags.join(", ")}
          onChange={(e) =>
            onChange({
              ...evalCase,
              tags: e.target.value.split(",").map((s) => s.trim()).filter(Boolean),
            })
          }
          placeholder="smoke, regression"
        />
      </div>
    </div>
  );
}
