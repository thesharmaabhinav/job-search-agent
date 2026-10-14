"""Tool definitions and executor for the Phase 8 tool-calling search agent."""
import json

from job_search import search_jobs as _search_jobs
from models import Job

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search for job postings. Returns how many jobs were found and how many are new. "
                "Use 'month' for date_posted if 'week' returns fewer than 5 results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Role or job title to search for, e.g. 'AI Engineer' or 'Data Scientist'.",
                    },
                    "location": {
                        "type": "string",
                        "description": "City, region, or country, e.g. 'Bangalore India' or 'Remote'.",
                    },
                    "date_posted": {
                        "type": "string",
                        "enum": ["today", "3days", "week", "month", "all"],
                        "description": "Recency filter. Defaults to 'week'.",
                    },
                },
                "required": ["query", "location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_results",
            "description": (
                "Returns a summary of all jobs collected so far in this session. "
                "Use to check coverage before deciding to search more or stop."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that you are satisfied with the collected jobs and the search is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why the search is complete.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


def execute_tool(name: str, args: dict, collected_jobs: list[Job]) -> tuple[str, bool]:
    """Execute a tool call. Mutates collected_jobs in place. Returns (result_text, is_done)."""
    if name == "finish":
        return f"Search complete. {args.get('reason', '')}", True

    if name == "get_previous_results":
        if not collected_jobs:
            return "No jobs collected yet.", False
        titles = ", ".join(f"{j.title} @ {j.company}" for j in collected_jobs[:10])
        tail = f" ... and {len(collected_jobs) - 10} more" if len(collected_jobs) > 10 else ""
        return f"{len(collected_jobs)} jobs collected: {titles}{tail}", False

    if name == "search_jobs":
        full_query = f"{args['query']} in {args['location']}"
        date_posted = args.get("date_posted", "week")
        jobs = _search_jobs(full_query, date_posted=date_posted)
        existing_urls = {j.url for j in collected_jobs}
        new_jobs = [j for j in jobs if j.url not in existing_urls]
        collected_jobs.extend(new_jobs)
        duplicates = len(jobs) - len(new_jobs)
        return (
            f"Found {len(jobs)} jobs ({len(new_jobs)} new, {duplicates} duplicates). "
            f"Total collected: {len(collected_jobs)}."
        ), False

    return f"Unknown tool: {name}", False
