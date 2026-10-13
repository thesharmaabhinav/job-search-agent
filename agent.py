import argparse
import json
import os
from datetime import date
from typing import TypedDict, Annotated
import operator
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from opik.integrations.openai import track_openai
import opik
from langgraph.graph import StateGraph, END

from models import CandidateProfile, Job, RankedJob
from cv_parser import parse_cv, PREFERRED_ROLES, SEARCH_LOCATIONS
from job_search import search_jobs
from ranker import rank_jobs

load_dotenv()

client = track_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

MIN_RESULTS_PER_QUERY = 5
MAX_RETRIES = 2


class AgentState(TypedDict):
    profile: CandidateProfile
    search_location: str
    pending_roles: list[str]
    current_role: str
    current_query: str
    raw_jobs: Annotated[list[Job], operator.add]
    ranked_jobs: list[RankedJob]
    query_result_count: int
    retry_count: int
    queries_made: int


def plan_query(state: AgentState) -> dict:
    role = state["pending_roles"][0]
    query = f"{role} in {state['search_location']}"
    return {
        "current_role": role,
        "current_query": query,
        "pending_roles": state["pending_roles"][1:],
        "retry_count": 0,
        "query_result_count": 0,
    }


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def rewrite_query(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a job search expert. Rewrite the given query to return more results by making it slightly broader. Return only the new query string, nothing else.",
            },
            {
                "role": "user",
                "content": f"Query '{state['current_query']}' returned too few results. Rewrite it to find more {state['current_role']} roles.",
            },
        ],
    )
    return {
        "current_query": response.choices[0].message.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }


def search(state: AgentState) -> dict:
    new_jobs = search_jobs(state["current_query"])
    return {
        "raw_jobs": new_jobs,
        "query_result_count": len(new_jobs),
        "queries_made": state["queries_made"] + 1,
    }


def route_after_search(state: AgentState) -> str:
    if state["query_result_count"] < MIN_RESULTS_PER_QUERY and state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    if state["pending_roles"]:
        return "plan_query"
    return "rank"


def rank(state: AgentState) -> dict:
    seen_urls: set[str] = set()
    unique_jobs: list[Job] = []
    for job in state["raw_jobs"]:
        if job.url not in seen_urls:
            seen_urls.add(job.url)
            unique_jobs.append(job)

    return {"ranked_jobs": rank_jobs(unique_jobs, state["profile"])}


def report(state: AgentState) -> dict:
    roles_searched = state["queries_made"] - (state["retry_count"])
    print(f"\n{'=' * 60}")
    print(f"Job Search Complete -- {len(state['ranked_jobs'])} unique jobs found")
    print(f"Location  : {state['search_location']}")
    print(f"Queries   : {state['queries_made']} total ({roles_searched} roles, {state['retry_count']} retries)")
    print(f"{'=' * 60}\n")
    for i, ranked_job in enumerate(state["ranked_jobs"], 1):
        job = ranked_job.job
        score = ranked_job.score
        print(f"{i}. {job.title} @ {job.company}")
        print(f"   Location : {job.location} (remote: {job.is_remote})")
        print(f"   Posted   : {job.posted_date or 'unknown'}")
        if job.salary_min:
            print(f"   Salary   : {job.salary_currency} {job.salary_min}--{job.salary_max}")
        print(
            f"   Scores   : skills={score.skills_match}/10  seniority={score.seniority_alignment}/10  "
            f"location={score.location_fit}/10  company={ranked_job.company_score}/10  "
            f"final={ranked_job.final_score:.3f}"
        )
        print(f"   Reason   : {score.reasoning}")
        print(f"   URL      : {job.url}")
        print()
    return {}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan_query", plan_query)
    graph.add_node("search", search)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("rank", rank)
    graph.add_node("report", report)

    graph.set_entry_point("plan_query")
    graph.add_edge("plan_query", "search")
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "rewrite_query": "rewrite_query",
            "plan_query": "plan_query",
            "rank": "rank",
        },
    )
    graph.add_edge("rewrite_query", "search")
    graph.add_edge("rank", "report")
    graph.add_edge("report", END)

    return graph.compile()


def _save_snapshot(jobs: list[Job]) -> None:
    seen: set[str] = set()
    unique = [j for j in jobs if not (j.url in seen or seen.add(j.url))]
    path = Path(__file__).parent / "jobs_snapshot.json"
    path.write_text(json.dumps([j.model_dump() for j in unique], indent=2))
    print(f"Snapshot saved: {len(unique)} unique jobs ({len(jobs) - len(unique)} duplicates removed) -> {path}")


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"), name="job-search-run")
def run_agent(profile: CandidateProfile, search_location: str, search_roles: list[str], save_snapshot: bool = False) -> None:
    opik.update_current_span(
        tags=[
            search_location,
            ", ".join(search_roles),
            date.today().isoformat(),
        ],
        metadata={
            "candidate_name": profile.name,
            "current_title": profile.current_title,
            "years_of_experience": profile.years_of_experience,
            "skills_count": len(profile.skills),
            "inferred_roles": profile.inferred_roles,
        },
    )

    final_state = build_graph().invoke({
        "profile": profile,
        "search_location": search_location,
        "pending_roles": search_roles,
        "current_role": "",
        "current_query": "",
        "raw_jobs": [],
        "ranked_jobs": [],
        "query_result_count": 0,
        "retry_count": 0,
        "queries_made": 0,
    })

    if save_snapshot:
        _save_snapshot(final_state["raw_jobs"])


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "resume" / "cv.pdf"
    profile = parse_cv(str(pdf_path))

    search_roles = PREFERRED_ROLES if PREFERRED_ROLES else profile.inferred_roles
    search_location = (
        SEARCH_LOCATIONS[0] if SEARCH_LOCATIONS
        else (profile.preferred_locations[0] if profile.preferred_locations else "India")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true", help="Save raw_jobs to jobs_snapshot.json after the run")
    args = parser.parse_args()

    run_agent(profile, search_location, search_roles, save_snapshot=args.snapshot)
