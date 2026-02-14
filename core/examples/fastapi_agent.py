"""
FastAPI Agent Server
-------------------
This module provides REST API endpoints for the Hive agent framework.
Allows executing agents, managing goals, and viewing execution results.

Run with:
    cd hive/core
    set PYTHONIOENCODING=utf-8
    set PYTHONPATH=core
    python -X utf8 examples/fastapi_agent.py

API will be available at http://localhost:8000
API docs at http://localhost:8000/docs
"""

import asyncio
import os
from datetime import datetime
from typing import Any

# Set API key for this example
os.environ.setdefault("ANTHROPIC_API_KEY", "csk-p3t9dj3r4rdkmjyfp3exvjkemjx48c6e5pwd8ychvm3e9jr9")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from framework.graph import EdgeCondition, EdgeSpec, Goal, GraphSpec, NodeSpec
from framework.graph.executor import ExecutionResult, GraphExecutor
from framework.runtime.core import Runtime


# Pydantic models for request/response
class SuccessCriterionCreate(BaseModel):
    id: str
    description: str
    metric: str
    target: str


class GoalCreate(BaseModel):
    id: str
    name: str
    description: str
    success_criteria: list[SuccessCriterionCreate] = Field(default_factory=list)


class NodeCreate(BaseModel):
    id: str
    name: str
    description: str
    node_type: str = "function"
    function: str | None = None
    input_keys: list[str] = Field(default_factory=list)
    output_keys: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    routes: dict[str, str] = Field(default_factory=dict)
    max_retries: int = 3


class EdgeCreate(BaseModel):
    id: str
    source: str
    target: str
    condition: str = "on_success"


class GraphCreate(BaseModel):
    id: str
    goal_id: str
    entry_node: str
    terminal_nodes: list[str] = Field(default_factory=list)
    nodes: list[NodeCreate]
    edges: list[EdgeCreate]


class ExecuteRequest(BaseModel):
    graph_id: str | None = None
    goal_id: str | None = None
    input_data: dict[str, Any] = Field(default_factory=dict)


class ExecutionResponse(BaseModel):
    success: bool
    output: dict[str, Any]
    error: str | None = None
    steps_executed: int = 0
    path: list[str] = Field(default_factory=list)
    paused_at: str | None = None


# Initialize FastAPI app
app = FastAPI(
    title="Hive Agent Framework API",
    description="REST API for the Hive agent framework",
    version="1.0.0",
)

# In-memory storage (in production, use a database)
goals: dict[str, Goal] = {}
graphs: dict[str, GraphSpec] = {}
function_registry: dict[str, callable] = {}
execution_history: list[ExecutionResult] = []


# Define default functions for the agent
def greet(name: str) -> str:
    """Generate a simple greeting."""
    return f"Hello, {name}!"


def uppercase(greeting: str) -> str:
    """Convert text to uppercase."""
    return greeting.upper()


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def reverse(text: str) -> str:
    """Reverse the text."""
    return text[::-1]


def extract_json(text: str) -> dict:
    """Extract JSON from text."""
    import json
    try:
        # Try to find JSON in the text
        start = text.find("{")
        if start >= 0:
            end = text.rfind("}") + 1
            if end > start:
                return json.loads(text[start:end])
    except:
        pass
    return {"error": "No valid JSON found"}


# Register default functions
function_registry["greet"] = greet
function_registry["uppercase"] = uppercase
function_registry["lowercase"] = lowercase
function_registry["reverse"] = reverse
function_registry["extract_json"] = extract_json


# Initialize default goal and graph
default_goal = Goal(
    id="greet-user",
    name="Greet User",
    description="Generate a friendly uppercase greeting",
    success_criteria=[
        {
            "id": "greeting_generated",
            "description": "Greeting produced",
            "metric": "custom",
            "target": "any",
        }
    ],
)
goals[default_goal.id] = default_goal

default_graph = GraphSpec(
    id="greeting-agent",
    goal_id="greet-user",
    entry_node="greeter",
    terminal_nodes=["uppercaser"],
    nodes=[
        NodeSpec(
            id="greeter",
            name="Greeter",
            description="Generates a simple greeting",
            node_type="function",
            function="greet",
            input_keys=["name"],
            output_keys=["greeting"],
        ),
        NodeSpec(
            id="uppercaser",
            name="Uppercaser",
            description="Converts greeting to uppercase",
            node_type="function",
            function="uppercase",
            input_keys=["greeting"],
            output_keys=["final_greeting"],
        ),
    ],
    edges=[
        EdgeSpec(
            id="greet-to-upper",
            source="greeter",
            target="uppercaser",
            condition=EdgeCondition.ON_SUCCESS,
        ),
    ],
)
graphs[default_graph.id] = default_graph


# Helper to get runtime and executor
def get_executor() -> GraphExecutor:
    runtime = Runtime(storage_path="./agent_logs")  # Use temp storage for API
    executor = GraphExecutor(runtime=runtime)

    # Register functions by node_id (not function name)
    # The graph uses node_id="greeter" with function="greet"
    # So we register by node_id so executor can find them
    node_func_map = {
        "greeter": greet,
        "uppercaser": uppercase,
        "lowercaser": lowercase,
        "reverser": reverse,
        "json_extractor": extract_json,
    }

    for node_id, func in node_func_map.items():
        executor.register_function(node_id, func)

    return executor


@app.get("/")
async def root():
    """Root endpoint - returns API info."""
    return {
        "name": "Hive Agent Framework API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "goals": "/goals",
            "graphs": "/graphs",
            "execute": "/execute",
            "history": "/history",
            "functions": "/functions",
        },
    }


@app.get("/goals")
async def list_goals():
    """List all available goals."""
    return {
        "goals": [
            {
                "id": g.id,
                "name": g.name,
                "description": g.description,
                "success_criteria": g.success_criteria,
            }
            for g in goals.values()
        ]
    }


@app.post("/goals")
async def create_goal(goal: GoalCreate):
    """Create a new goal."""
    if goal.id in goals:
        raise HTTPException(status_code=400, detail=f"Goal '{goal.id}' already exists")

    # Convert to dict format for Goal
    criteria = [
        {
            "id": c.id,
            "description": c.description,
            "metric": c.metric,
            "target": c.target,
        }
        for c in goal.success_criteria
    ]

    new_goal = Goal(
        id=goal.id,
        name=goal.name,
        description=goal.description,
        success_criteria=criteria,
    )
    goals[goal.id] = new_goal
    return {"id": new_goal.id, "name": new_goal.name}


@app.get("/goals/{goal_id}")
async def get_goal(goal_id: str):
    """Get a specific goal by ID."""
    if goal_id not in goals:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")

    goal = goals[goal_id]
    return {
        "id": goal.id,
        "name": goal.name,
        "description": goal.description,
        "success_criteria": goal.success_criteria,
    }


@app.get("/graphs")
async def list_graphs():
    """List all available graphs."""
    return {
        "graphs": [
            {
                "id": g.id,
                "goal_id": g.goal_id,
                "entry_node": g.entry_node,
                "terminal_nodes": g.terminal_nodes,
                "node_count": len(g.nodes),
                "edge_count": len(g.edges),
            }
            for g in graphs.values()
        ]
    }


@app.post("/graphs")
async def create_graph(graph: GraphCreate):
    """Create a new graph."""
    if graph.id in graphs:
        raise HTTPException(status_code=400, detail=f"Graph '{graph.id}' already exists")

    # Validate goal exists
    if graph.goal_id not in goals:
        raise HTTPException(status_code=400, detail=f"Goal '{graph.goal_id}' not found")

    # Convert nodes
    nodes = []
    for node in graph.nodes:
        nodes.append(NodeSpec(
            id=node.id,
            name=node.name,
            description=node.description,
            node_type=node.node_type,
            function=node.function,
            input_keys=node.input_keys,
            output_keys=node.output_keys,
            tools=node.tools,
            system_prompt=node.system_prompt,
            routes=node.routes,
            max_retries=node.max_retries,
        ))

    # Convert edges
    edges = []
    for edge in graph.edges:
        condition = EdgeCondition.ON_SUCCESS
        if edge.condition == "on_failure":
            condition = EdgeCondition.ON_FAILURE
        elif edge.condition == "always":
            condition = EdgeCondition.ALWAYS

        edges.append(EdgeSpec(
            id=edge.id,
            source=edge.source,
            target=edge.target,
            condition=condition,
        ))

    new_graph = GraphSpec(
        id=graph.id,
        goal_id=graph.goal_id,
        entry_node=graph.entry_node,
        terminal_nodes=graph.terminal_nodes,
        nodes=nodes,
        edges=edges,
    )

    graphs[graph.id] = new_graph
    return {"id": new_graph.id, "node_count": len(new_graph.nodes)}


@app.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    """Get a specific graph by ID."""
    if graph_id not in graphs:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    graph = graphs[graph_id]
    return {
        "id": graph.id,
        "goal_id": graph.goal_id,
        "entry_node": graph.entry_node,
        "terminal_nodes": graph.terminal_nodes,
        "nodes": [
            {
                "id": n.id,
                "name": n.name,
                "description": n.description,
                "node_type": n.node_type,
                "function": n.function,
                "input_keys": n.input_keys,
                "output_keys": n.output_keys,
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "condition": e.condition.value,
            }
            for e in graph.edges
        ],
    }


@app.get("/functions")
async def list_functions():
    """List all registered functions."""
    return {
        "functions": list(function_registry.keys())
    }


@app.post("/functions/{func_name}")
async def register_function(func_name: str, code: str):
    """Register a new function from Python code."""
    try:
        # Create a function from the code
        local_ns = {}
        exec(code, {}, local_ns)

        # Find the function in local namespace
        func = local_ns.get(func_name)
        if func is None:
            raise HTTPException(status_code=400, detail=f"Function '{func_name}' not found in code")

        function_registry[func_name] = func
        return {"status": "registered", "function": func_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to register function: {str(e)}")


@app.post("/execute", response_model=ExecutionResponse)
async def execute_agent(request: ExecuteRequest):
    """Execute an agent with the given input data."""
    # Determine which graph and goal to use
    graph_id = request.graph_id or "greeting-agent"
    goal_id = request.goal_id or "greet-user"

    if graph_id not in graphs:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    if goal_id not in goals:
        raise HTTPException(status_code=404, detail=f"Goal '{goal_id}' not found")

    graph = graphs[graph_id]
    goal = goals[goal_id]

    # Get executor and execute
    executor = get_executor()

    try:
        result = await executor.execute(
            graph=graph,
            goal=goal,
            input_data=request.input_data,
        )

        # Store in history
        execution_history.append(result)

        return ExecutionResponse(
            success=result.success,
            output=result.output,
            error=result.error,
            steps_executed=result.steps_executed,
            path=result.path,
            paused_at=result.paused_at,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.get("/history")
async def get_history(limit: int = 10):
    """Get execution history."""
    return {
        "executions": [
            {
                "success": e.success,
                "output": e.output,
                "error": e.error,
                "steps_executed": e.steps_executed,
                "path": e.path,
            }
            for e in execution_history[-limit:]
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    print("Starting Hive Agent Framework API...")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
