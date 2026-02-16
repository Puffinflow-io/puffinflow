import { create } from "zustand";
import type { WorkflowNode, WorkflowEdge, NodeConfig, Position, WorkflowIR } from "@/lib/types";
import { NodeType } from "@/lib/types";
import { getDefaultConfig } from "@/lib/nodeRegistry";

interface WorkflowState {
  // Data
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  selectedNodeId: string | null;
  yamlContent: string;
  pythonCode: string;
  isDirty: boolean;
  projectId: string | null;
  workflowId: string | null;
  workflowName: string;

  // Actions
  addNode: (type: NodeType, position: Position) => string;
  removeNode: (id: string) => void;
  updateNodeConfig: (id: string, config: NodeConfig) => void;
  updateNodePosition: (id: string, position: Position) => void;
  addEdge: (from_node: string, to_node: string, label?: string) => void;
  removeEdge: (from_node: string, to_node: string) => void;
  setSelectedNode: (id: string | null) => void;
  setPythonCode: (code: string) => void;
  setDirty: (dirty: boolean) => void;
  setWorkflowName: (name: string) => void;
  loadFromIR: (ir: WorkflowIR) => void;
  serializeToIR: () => WorkflowIR;
  reset: () => void;
}

let nodeCounter = 0;

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  yamlContent: "",
  pythonCode: "",
  isDirty: false,
  projectId: null,
  workflowId: null,
  workflowName: "Untitled Workflow",

  addNode: (type: NodeType, position: Position) => {
    const id = `${type}_${++nodeCounter}`;
    const defaultConfig = getDefaultConfig(type);
    const newNode: WorkflowNode = {
      id,
      type,
      position,
      config: defaultConfig as NodeConfig,
    };
    set((state) => ({
      nodes: [...state.nodes, newNode],
      isDirty: true,
    }));
    return id;
  },

  removeNode: (id: string) => {
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.from_node !== id && e.to_node !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
      isDirty: true,
    }));
  },

  updateNodeConfig: (id: string, config: NodeConfig) => {
    set((state) => ({
      nodes: state.nodes.map((n) => (n.id === id ? { ...n, config } : n)),
      isDirty: true,
    }));
  },

  updateNodePosition: (id: string, position: Position) => {
    set((state) => ({
      nodes: state.nodes.map((n) => (n.id === id ? { ...n, position } : n)),
    }));
  },

  addEdge: (from_node: string, to_node: string, label?: string) => {
    set((state) => {
      const exists = state.edges.some(
        (e) => e.from_node === from_node && e.to_node === to_node
      );
      if (exists) return state;
      return {
        edges: [...state.edges, { from_node, to_node, label }],
        isDirty: true,
      };
    });
  },

  removeEdge: (from_node: string, to_node: string) => {
    set((state) => ({
      edges: state.edges.filter(
        (e) => !(e.from_node === from_node && e.to_node === to_node)
      ),
      isDirty: true,
    }));
  },

  setSelectedNode: (id: string | null) => set({ selectedNodeId: id }),
  setPythonCode: (code: string) => set({ pythonCode: code }),
  setDirty: (dirty: boolean) => set({ isDirty: dirty }),
  setWorkflowName: (name: string) => set({ workflowName: name, isDirty: true }),

  loadFromIR: (ir: WorkflowIR) => {
    set({
      nodes: ir.nodes,
      edges: ir.edges,
      workflowName: ir.metadata.name,
      isDirty: false,
      selectedNodeId: null,
    });
  },

  serializeToIR: (): WorkflowIR => {
    const state = get();
    return {
      version: "1.0",
      metadata: {
        name: state.workflowName,
        description: "",
        author: "",
        tags: [],
      },
      agent: {
        name: state.workflowName.toLowerCase().replace(/\s+/g, "_"),
        class_name: state.workflowName.replace(/\s+/g, ""),
        max_concurrent: 5,
      },
      inputs: [],
      outputs: [],
      nodes: state.nodes,
      edges: state.edges,
    };
  },

  reset: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      yamlContent: "",
      pythonCode: "",
      isDirty: false,
      projectId: null,
      workflowId: null,
      workflowName: "Untitled Workflow",
    }),
}));
