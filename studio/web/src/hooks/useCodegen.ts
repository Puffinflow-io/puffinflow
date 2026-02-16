"use client";
import { useEffect, useRef, useState, useCallback } from "react";

interface UseCodegenOptions {
  debounceMs?: number;
}

export function useCodegen(options: UseCodegenOptions = {}) {
  const { debounceMs = 300 } = options;
  const [pythonCode, setPythonCode] = useState<string>("");
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/codegen/ws/live-preview`);

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.python) {
          setPythonCode(data.python);
        }
        if (data.error) {
          setError(data.error);
        }
      } catch {
        setPythonCode(event.data);
      }
    };

    ws.onerror = () => {
      setError("WebSocket connection error");
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const sendYaml = useCallback(
    (yaml: string) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ yaml }));
        }
      }, debounceMs);
    },
    [debounceMs]
  );

  return { pythonCode, isConnected, error, sendYaml };
}
