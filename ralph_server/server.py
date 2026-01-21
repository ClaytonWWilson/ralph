#!/usr/bin/env python
"""
MCP Server for Ralph - Task Management over stdio

This server exposes PRD task management tools via the Model Context Protocol.
AI agents connect to this server to list, start, complete, and fail tasks.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Parse command-line arguments before creating the server
parser = argparse.ArgumentParser(description="Ralph MCP Server")
parser.add_argument(
    "--prd-path",
    type=str,
    default="./prds.json",
    help="Path to the PRD JSON file (default: ./prds.json)",
)
parser.add_argument(
    "--log-file",
    type=str,
    default="ralph_server.log",
    help="Path to log file (default: ralph_server.log)",
)
args, _ = parser.parse_known_args()

# Setup logging to file
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Create a task completion log file
task_log_file = Path("ralph_task_completions.log")

# Create an MCP server
mcp = FastMCP(
    "Ralph Wiggum",
    instructions="View and manage pending project tasks. Use list_tasks() to see tasks, start_task() to begin work, complete_task() to mark done, or fail_task() if blocked.",
)


@dataclass
class ServerState:
    """Maintains session state within the MCP server."""

    prd_path: Path
    session_start_time: float = field(
        default_factory=lambda: datetime.now().timestamp()
    )
    current_task_id: Optional[str] = None
    iteration_count: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0


# Global server state
state = ServerState(prd_path=Path(args.prd_path))


def load_prd() -> dict:
    """Load the full PRD file."""
    if not state.prd_path.exists():
        return {"tasks": []}
    return json.loads(state.prd_path.read_text())


def save_prd(prd_data: dict) -> None:
    """Save the PRD file atomically."""
    temp_path = state.prd_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(prd_data, indent=2))
    temp_path.replace(state.prd_path)


def load_pending_tasks() -> list[dict]:
    """Load only incomplete tasks from the PRD."""
    prd_data = load_prd()
    tasks = prd_data.get("tasks", prd_data.get("features", prd_data.get("items", [])))

    pending_tasks = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        # Check various status field names
        status = t.get("passes", False)
        if not status:
            pending_tasks.append(t)

    return pending_tasks


def get_task_by_id(task_id: str) -> Optional[dict]:
    """Get a specific task by ID."""
    prd_data = load_prd()
    tasks = prd_data.get("tasks", prd_data.get("features", prd_data.get("items", [])))

    for t in tasks:
        if isinstance(t, dict) and t.get("id") == task_id:
            return t
    return None


def update_task_in_prd(task_id: str, updates: dict) -> bool:
    """Update a task's fields in the PRD file."""
    prd_data = load_prd()
    tasks = prd_data.get("tasks", prd_data.get("features", prd_data.get("items", [])))

    for t in tasks:
        if isinstance(t, dict) and t.get("id") == task_id:
            t.update(updates)
            save_prd(prd_data)
            return True
    return False


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
def list_tasks() -> list[dict]:
    """
    List all tasks that are not yet completed.

    Returns a list of task objects with their id, category, description, and steps.
    Use this to see what work needs to be done.
    """
    logger.info("list_tasks called")
    tasks = load_pending_tasks()
    logger.info(f"list_tasks returning {len(tasks)} pending tasks")
    return tasks


@mcp.tool()
def get_task(task_id: str) -> dict:
    """
    Get detailed information about a specific task.

    Args:
        task_id: The unique identifier of the task to retrieve.

    Returns:
        The full task object including all fields.

    Raises:
        ValueError: If the task is not found.
    """
    logger.info(f"get_task called: {task_id}")
    task = get_task_by_id(task_id)
    if task is None:
        logger.error(f"get_task failed: task '{task_id}' not found")
        raise ValueError(f"Task '{task_id}' not found")
    logger.info(f"get_task returning task: {task_id}")
    return task


@mcp.tool()
def start_task(task_id: str) -> dict:
    """
    Mark a task as in progress and set it as the current active task.

    Call this when you begin working on a task. This helps track which task
    is currently being worked on.

    Args:
        task_id: The unique identifier of the task to start.

    Returns:
        The updated task object.

    Raises:
        ValueError: If the task is not found.
    """
    logger.info(f"start_task called: {task_id}")

    task = get_task_by_id(task_id)
    if task is None:
        logger.error(f"start_task failed: task '{task_id}' not found")
        raise ValueError(f"Task '{task_id}' not found")

    # Update task status
    updates = {
        "status": "in_progress",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    update_task_in_prd(task_id, updates)

    # Update server state
    state.current_task_id = task_id
    logger.info(f"start_task: marked {task_id} as in_progress")

    # Log the start to a file for ralph_mcp.py to read
    try:
        with open(task_log_file, "a") as f:
            f.write(
                f"<time>{datetime.now().isoformat()}</time><start>{task_id}</start>\n"
            )
    except Exception as e:
        logger.error(f"Failed to write to task log: {e}")

    # Return updated task
    updated_task = get_task_by_id(task_id)
    assert updated_task is not None  # We just verified it exists
    return updated_task


@mcp.tool()
def complete_task(task_id: str, notes: str | None = None) -> dict:
    """
    Mark a task as completed/passing.

    Call this after you have successfully implemented and tested the task.
    This will update the PRD to mark the task as passing.

    Args:
        task_id: The unique identifier of the task to complete.
        notes: Optional notes about the completion (e.g., what was implemented).

    Returns:
        The updated task object.

    Raises:
        ValueError: If the task is not found.
    """
    logger.info(f"complete_task called: {task_id}, notes={notes}")

    task = get_task_by_id(task_id)
    if task is None:
        logger.error(f"complete_task failed: task '{task_id}' not found")
        raise ValueError(f"Task '{task_id}' not found")

    # Update task status
    updates = {
        "passes": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if notes:
        updates["completion_notes"] = notes

    update_task_in_prd(task_id, updates)

    # Update server state
    state.tasks_completed += 1
    if state.current_task_id == task_id:
        state.current_task_id = None

    logger.info(
        f"complete_task: marked {task_id} as completed (total completed this session: {state.tasks_completed})"
    )

    # Log the complete to a file for ralph_mcp.py to read
    try:
        with open(task_log_file, "a") as f:
            f.write(
                f"<time>{datetime.now().isoformat()}</time><complete>{task_id}</complete>\n"
            )
    except Exception as e:
        logger.error(f"Failed to write to task log: {e}")

    # Return updated task
    updated_task = get_task_by_id(task_id)
    assert updated_task is not None  # We just verified it exists
    return updated_task


@mcp.tool()
def fail_task(task_id: str, reason: str) -> dict:
    """
    Mark a task as failed with a reason.

    Call this when you cannot complete a task due to blockers, errors, or
    other issues. The task status will NOT be updated to passing, but the
    failure will be recorded.

    Args:
        task_id: The unique identifier of the task that failed.
        reason: A description of why the task failed (blockers, errors, etc.).

    Returns:
        The task object with failure information.

    Raises:
        ValueError: If the task is not found.
    """
    logger.info(f"fail_task called: {task_id}, reason={reason}")

    task = get_task_by_id(task_id)
    if task is None:
        logger.error(f"fail_task failed: task '{task_id}' not found")
        raise ValueError(f"Task '{task_id}' not found")

    # Update task with failure info (but don't mark as passing)
    updates = {
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "failure_reason": reason,
    }
    update_task_in_prd(task_id, updates)

    # Update server state
    state.tasks_failed += 1
    if state.current_task_id == task_id:
        state.current_task_id = None

    logger.info(
        f"fail_task: marked {task_id} as failed (total failed this session: {state.tasks_failed})"
    )

    # Log the fail to a file for ralph_mcp.py to read
    try:
        with open(task_log_file, "a") as f:
            f.write(
                f"<time>{datetime.now().isoformat()}</time><fail>{task_id}</fail>\n"
            )
    except Exception as e:
        logger.error(f"Failed to write to task log: {e}")

    # Return updated task
    updated_task = get_task_by_id(task_id)
    assert updated_task is not None  # We just verified it exists
    return updated_task


@mcp.tool()
def get_progress() -> dict:
    """
    Get current session progress statistics.

    Returns information about the PRD progress, session duration,
    and task completion stats.

    Returns:
        A dictionary with progress information including:
        - total_tasks: Total number of tasks in the PRD
        - completed_tasks: Number of tasks marked as passing
        - remaining_tasks: Number of incomplete tasks
        - session_duration: Time since session started (seconds)
        - current_task: ID of the currently active task (if any)
        - tasks_completed_this_session: Tasks completed in this session
        - tasks_failed_this_session: Tasks failed in this session
    """
    logger.info("get_progress called")

    prd_data = load_prd()
    tasks = prd_data.get("tasks", prd_data.get("features", prd_data.get("items", [])))

    total = 0
    passing = 0
    categories = {}

    for t in tasks:
        if not isinstance(t, dict):
            continue
        total += 1

        status = t.get("passes", False)

        if status:
            passing += 1

        # Track by category
        category = t.get("category", "uncategorized")
        if category not in categories:
            categories[category] = {"total": 0, "passing": 0}
        categories[category]["total"] += 1
        if status:
            categories[category]["passing"] += 1

    return {
        "total_tasks": total,
        "completed_tasks": passing,
        "remaining_tasks": total - passing,
        "completion_percentage": (passing / total * 100) if total > 0 else 0,
        "session_duration": datetime.now().timestamp() - state.session_start_time,
        "current_task": state.current_task_id,
        "tasks_completed_this_session": state.tasks_completed,
        "tasks_failed_this_session": state.tasks_failed,
        "categories": categories,
    }


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("prd://tasks")
def resource_all_tasks() -> str:
    """
    Full PRD task list as JSON.

    Returns all tasks from the PRD file, both complete and incomplete.
    """
    prd_data = load_prd()
    tasks = prd_data.get("tasks", prd_data.get("features", prd_data.get("items", [])))
    return json.dumps(tasks, indent=2)


@mcp.resource("prd://current")
def resource_current_task() -> str:
    """
    Currently active task as JSON.

    Returns the task currently being worked on, or null if none.
    """
    if state.current_task_id is None:
        return json.dumps(None)

    task = get_task_by_id(state.current_task_id)
    return json.dumps(task, indent=2)


# Run with stdio transport
if __name__ == "__main__":
    logger.info(
        f"Starting Ralph MCP server with PRD: {state.prd_path}, log file: {args.log_file}"
    )
    mcp.run(transport="stdio")
