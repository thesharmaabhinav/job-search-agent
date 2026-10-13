import os
import requests
from dotenv import load_dotenv
import opik

from models import Job

load_dotenv()

_JSEARCH_HOST = "jsearch.p.rapidapi.com"
_HEADERS = {
    "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
    "X-RapidAPI-Host": _JSEARCH_HOST,
}


def _normalise(raw: dict) -> Job:
    location_parts = [
        p for p in [
            raw.get("job_city") or "",
            raw.get("job_state") or "",
            raw.get("job_country") or "",
        ]
        if p
    ]
    return Job(
        title=raw.get("job_title", ""),
        company=raw.get("employer_name", ""),
        location=", ".join(location_parts),
        is_remote=raw.get("job_is_remote", False),
        description=raw.get("job_description", ""),
        employment_type=raw.get("job_employment_type"),
        salary_min=raw.get("job_min_salary"),
        salary_max=raw.get("job_max_salary"),
        salary_currency=raw.get("job_salary_currency"),
        url=raw.get("job_apply_link", ""),
        posted_date=raw.get("job_posted_at_datetime_utc"),
    )


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def search_jobs(query: str, num_pages: int = 1, date_posted: str = "week") -> list[Job]:
    response = requests.get(
        f"https://{_JSEARCH_HOST}/search-v2",
        headers=_HEADERS,
        params={"query": query, "page": 1, "num_pages": num_pages, "date_posted": date_posted},
    )
    response.raise_for_status()
    return [_normalise(job) for job in response.json().get("data", {}).get("jobs", [])]


if __name__ == "__main__":
    jobs = search_jobs("AI Engineer in Bangalore India")
    for job in jobs:
        print(f"{job.title} @ {job.company} — {job.location} (remote: {job.is_remote})")
        print(f"  Posted : {job.posted_date or 'unknown'}")
        if job.salary_min:
            print(f"  Salary : {job.salary_currency} {job.salary_min}–{job.salary_max}")
        print(f"  URL    : {job.url}")
        print()
