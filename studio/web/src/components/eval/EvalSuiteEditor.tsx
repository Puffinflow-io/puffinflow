"use client";
import { useState } from "react";
import { Input } from "@/components/ui/Input";
import { Label } from "@/components/ui/Label";
import { Button } from "@/components/ui/Button";
import { Textarea } from "@/components/ui/Textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import type { EvalSuite } from "@/lib/types";

interface EvalSuiteEditorProps {
  suite: EvalSuite;
  onSave: (data: { name: string; scoring_config: Record<string, unknown> }) => void;
}

export default function EvalSuiteEditor({ suite, onSave }: EvalSuiteEditorProps) {
  const [name, setName] = useState(suite.name);
  const [scoringConfigStr, setScoringConfigStr] = useState(
    JSON.stringify(suite.scoring_config || {}, null, 2)
  );
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      let parsedConfig: Record<string, unknown> = {};
      try {
        parsedConfig = JSON.parse(scoringConfigStr);
      } catch {
        // Keep empty config if invalid JSON
      }
      onSave({ name, scoring_config: parsedConfig });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Suite Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label htmlFor="suite-name">Suite Name</Label>
          <Input
            id="suite-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="scoring-config">Scoring Config (JSON)</Label>
          <Textarea
            id="scoring-config"
            value={scoringConfigStr}
            onChange={(e) => setScoringConfigStr(e.target.value)}
            rows={4}
            className="font-mono text-xs"
          />
        </div>
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? "Saving..." : "Save Configuration"}
        </Button>
      </CardContent>
    </Card>
  );
}
