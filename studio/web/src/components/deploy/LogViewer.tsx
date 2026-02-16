"use client";
import { useEffect, useRef } from "react";

interface LogViewerProps {
  logs: string;
  className?: string;
}

export default function LogViewer({ logs, className }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div
      ref={containerRef}
      className={`bg-gray-950 text-gray-100 rounded-lg border border-border p-4 h-80 overflow-y-auto ${className || ""}`}
    >
      <pre className="text-xs font-mono whitespace-pre-wrap">
        {logs || "No logs available."}
      </pre>
    </div>
  );
}
