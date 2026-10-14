import argparse
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from opik.integrations.openai import track_openai
import opik

from models import CandidateProfile, Job, RankedJob
from cv_parser import parse_cv, PREFERRED_ROLES, SEARCH_LOCATIONS
from ranker import rank_jobs
from tools import TOOLS, execute_tool

load_dotenv()

client = track_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

MAX_TOOL_CALLS = 15


def _build_system_prompt(
    profile: CandidateProfile,
    preferred_roles: list[str] | None,
    preferred_locations: list[str] | None,
) -> str:
    roles = ", ".join(preferred_roles or profile.inferred_roles)
    locs_list = preferred_locations or profile.preferred_locations
    locations = ", ".join(locs_list) if locs_list else "flexible"
    skills = ", ".join(profile.skills[:15])
    remote = "open to remote" if profile.open_to_remote else "not looking for remote"

    return (
        f"You are a job search agent for {profile.name}, "
        f"a {profile.current_title} with {profile.years_of_experience} years of experience.\n\n"
        f"Candidate summary:\n"
        f"- Target roles: {roles}\n"
        f"- Key skills: {skills}\n"
        f"- Preferred locations: {locations}\n"
        f"- Remote: {remote}\n\n"
        f"Your goal is to collect a diverse, high-quality set of job postings "
        f"(aim for 20-40 jobs) relevant to this candidate.\n\n"
        f"Strategy:\n"
        f"- Search across the candidate's target roles and preferred locations\n"
        f"- If a query returns very few results, try broadening it or extending date_posted to 'month'\n"
        f"- Avoid repeating the same role + location combination you already searched\n"
        f"- Once you have good coverage across roles and locations, call finish()\n\n"
        f"You have a budget of {MAX_TOOL_CALLS} tool calls total. Use them wisely."
    )


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def run_search_loop(
    profile: CandidateProfile,
    preferred_roles: list[str] | None,
    preferred_locations: list[str] | None,
) -> tuple[list[Job], int]:
    """Run the tool-calling search loop. Returns (collected_jobs, tool_calls_used)."""
    collected_jobs: list[Job] = []
    tool_call_count = 0
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(profile, preferred_roles, preferred_locations)}
    ]

    while tool_call_count < MAX_TOOL_CALLS:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Append assistant turn as a plain dict for reliable re-serialisation
        assistant_entry: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_entry)

        if not message.tool_calls:
            break  # LLM finished without calling a tool

        done = False
        for tc in message.tool_calls:
            tool_call_count += 1
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            args_display = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"[{tool_call_count}/{MAX_TOOL_CALLS}] {tc.function.name}({args_display})")

            result, is_done = execute_tool(tc.function.name, args, collected_jobs)
            print(f"  -> {result}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if is_done:
                done = True

        if done:
            break

    print(f"\nSearch complete. {len(collected_jobs)} jobs collected in {tool_call_count}/{MAX_TOOL_CALLS} tool calls.")
    print("Ranking jobs...")
    return collected_jobs, tool_call_count


def report(ranked_jobs: list[RankedJob], tool_calls_used: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"Job Search Complete -- {len(ranked_jobs)} jobs ranked")
    print(f"Tool calls used    : {tool_calls_used} / {MAX_TOOL_CALLS}")
    print(f"{'=' * 60}\n")
    for i, ranked_job in enumerate(ranked_jobs, 1):
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


def _save_snapshot(jobs: list[Job]) -> None:
    seen: set[str] = set()
    unique = [j for j in jobs if not (j.url in seen or seen.add(j.url))]
    path = Path(__file__).parent / "jobs_snapshot.json"
    path.write_text(json.dumps([j.model_dump() for j in unique], indent=2))
    print(f"Snapshot saved: {len(unique)} unique jobs ({len(jobs) - len(unique)} duplicates removed) -> {path}")


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"), name="job-search-run")
def run_agent(
    profile: CandidateProfile,
    save_snapshot: bool = False,
    preferred_roles: list[str] | None = None,
    preferred_locations: list[str] | None = None,
) -> None:
    opik.update_current_span(
        tags=[
            ", ".join(preferred_roles or profile.inferred_roles),
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

    collected_jobs, tool_calls_used = run_search_loop(profile, preferred_roles, preferred_locations)

    if save_snapshot:
        _save_snapshot(collected_jobs)

    ranked = rank_jobs(collected_jobs, profile)
    report(ranked, tool_calls_used)


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "resume" / "cv.pdf"
    profile = parse_cv(str(pdf_path))

    preferred_roles = PREFERRED_ROLES if PREFERRED_ROLES else None
    preferred_locations = SEARCH_LOCATIONS if SEARCH_LOCATIONS else None

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true", help="Save raw jobs to jobs_snapshot.json after the run")
    args = parser.parse_args()

    run_agent(profile, save_snapshot=args.snapshot, preferred_roles=preferred_roles, preferred_locations=preferred_locations)
