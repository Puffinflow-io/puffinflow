"use client";
import { useCallback, useMemo, useRef, type DragEvent } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge as RFEdge,
  type Node as RFNode,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useWorkflowStore } from "@/stores/workflowStore";
import { NodeType } from "@/lib/types";
import { getNodeTypeInfo } from "@/lib/nodeRegistry";
import {
  LLMNode,
  FunctionNode,
  ConditionalNode,
  InputNode,
  OutputNode,
  SubgraphNode,
  ToolNode,
  MemoryNode,
  FanOutNode,
  MergeNode,
} from "@/components/nodes";
import { useEffect } from "react";

const nodeTypes = {
  [NodeType.LLM]: LLMNode,
  [NodeType.FUNCTION]: FunctionNode,
  [NodeType.CONDITIONAL]: ConditionalNode,
  [NodeType.INPUT]: InputNode,
  [NodeType.OUTPUT]: OutputNode,
  [NodeType.SUBGRAPH]: SubgraphNode,
  [NodeType.TOOL]: ToolNode,
  [NodeType.MEMORY]: MemoryNode,
  [NodeType.FAN_OUT]: FanOutNode,
  [NodeType.MERGE]: MergeNode,
};

function toRFNodes(storeNodes: ReturnType<typeof useWorkflowStore.getState>["nodes"]): RFNode[] {
  return storeNodes.map((n) => ({
    id: n.id,
    type: n.type,
    position: n.position,
    data: { config: n.config, label: getNodeTypeInfo(n.type).label },
    selected: false,
  }));
}

function toRFEdges(storeEdges: ReturnType<typeof useWorkflowStore.getState>["edges"]): RFEdge[] {
  return storeEdges.map((e, i) => ({
    id: `edge-${e.from_node}-${e.to_node}-${i}`,
    source: e.from_node,
    target: e.to_node,
    label: e.label,
    animated: true,
    style: { stroke: "#6366f1" },
  }));
}

function CanvasInner() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const storeNodes = useWorkflowStore((s) => s.nodes);
  const storeEdges = useWorkflowStore((s) => s.edges);
  const addNodeToStore = useWorkflowStore((s) => s.addNode);
  const addEdgeToStore = useWorkflowStore((s) => s.addEdge);
  const updateNodePosition = useWorkflowStore((s) => s.updateNodePosition);
  const setSelectedNode = useWorkflowStore((s) => s.setSelectedNode);
  const removeNodeFromStore = useWorkflowStore((s) => s.removeNode);
  const removeEdgeFromStore = useWorkflowStore((s) => s.removeEdge);

  const initialNodes = useMemo(() => toRFNodes(storeNodes), [storeNodes]);
  const initialEdges = useMemo(() => toRFEdges(storeEdges), [storeEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => {
    setNodes(toRFNodes(storeNodes));
  }, [storeNodes, setNodes]);

  useEffect(() => {
    setEdges(toRFEdges(storeEdges));
  }, [storeEdges, setEdges]);

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds));
      if (params.source && params.target) {
        addEdgeToStore(params.source, params.target);
      }
    },
    [setEdges, addEdgeToStore]
  );

  const onNodeDragStop = useCallback(
    (_: unknown, node: RFNode) => {
      updateNodePosition(node.id, node.position);
    },
    [updateNodePosition]
  );

  const onNodeClick = useCallback(
    (_: unknown, node: RFNode) => {
      setSelectedNode(node.id);
    },
    [setSelectedNode]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  const onNodesDelete = useCallback(
    (deleted: RFNode[]) => {
      deleted.forEach((n) => removeNodeFromStore(n.id));
    },
    [removeNodeFromStore]
  );

  const onEdgesDelete = useCallback(
    (deleted: RFEdge[]) => {
      deleted.forEach((e) => removeEdgeFromStore(e.source, e.target));
    },
    [removeEdgeFromStore]
  );

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      const type = event.dataTransfer.getData("application/reactflow") as NodeType;
      if (!type) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addNodeToStore(type, position);
    },
    [screenToFlowPosition, addNodeToStore]
  );

  return (
    <div ref={reactFlowWrapper} className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDragStop={onNodeDragStop}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onNodesDelete={onNodesDelete}
        onEdgesDelete={onEdgesDelete}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        fitView
        className="bg-background"
      >
        <Background gap={16} size={1} />
        <Controls />
        <MiniMap
          nodeStrokeWidth={3}
          className="!bg-muted"
        />
      </ReactFlow>
    </div>
  );
}

export default function WorkflowCanvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
