"""Re-score and rank snapshot jobs using the current ranker config.

Usage:
  uv run python rerank.py                  # fresh LLM scoring of all snapshot jobs
  uv run python rerank.py --reweight-only  # recompute final_score from stored scores, no LLM calls
"""
import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cv_parser import parse_cv
from models import Job
from ranker import RANKER_CONFIG, score_jobs_for_eval

SNAPSHOT_PATH = Path(__file__).parent / "jobs_snapshot.json"
RANKED_JOBS_PATH = Path(__file__).parent / "ranked_jobs.json"


def _compute_final_score(job: dict) -> float:
    base = (
        (job["skills_match"] / 10) * RANKER_CONFIG["w_skills"]
        + (job["seniority_alignment"] / 10) * RANKER_CONFIG["w_seniority"]
        + (job["location_fit"] / 10) * RANKER_CONFIG["w_location"]
        + (job["company_score"] / 10) * RANKER_CONFIG["w_company"]
    )
    sf = job.get("salary_fit")
    salary_bonus = (sf / 10) * RANKER_CONFIG["w_salary_bonus"] if sf is not None else 0.0
    return round(base + salary_bonus, 4)


def reweight_only() -> None:
    """Recompute final_score from existing dimension scores. No LLM calls."""
    if not RANKED_JOBS_PATH.exists():
        print("No ranked_jobs.json found. Run without --reweight-only to score from scratch.")
        return

    jobs = json.loads(RANKED_JOBS_PATH.read_text())
    print(f"Reweighting {len(jobs)} jobs with current config...")

    for job in jobs:
        job["final_score"] = _compute_final_score(job)

    jobs.sort(key=lambda j: j["final_score"], reverse=True)
    RANKED_JOBS_PATH.write_text(json.dumps(jobs, indent=2))
    print(f"Done. Re-sorted {len(jobs)} jobs by new weights.")


def fresh_score() -> None:
    """Score all snapshot jobs with LLM judge, save to ranked_jobs.json with reasoning."""
    if not SNAPSHOT_PATH.exists():
        print("No snapshot found. Run: uv run python agent.py --snapshot")
        return

    jobs_raw = json.loads(SNAPSHOT_PATH.read_text())
    jobs = [Job(**j) for j in jobs_raw]
    print(f"Loaded {len(jobs)} jobs from snapshot.")

    pdf_path = Path(__file__).parent / "resume" / "cv.pdf"
    profile = parse_cv(str(pdf_path))
    print(f"Parsed CV: {profile.name}")

    config_key = hashlib.md5(
        (RANKER_CONFIG["judge_prompt"] + RANKER_CONFIG["llm_model"]).encode()
    ).hexdigest()[:8]
    print(f"Scoring {len(jobs)} jobs via LLM judge (config_key={config_key})...")

    scored = score_jobs_for_eval(jobs, profile)

    output = []
    for ranked in scored:
        job = ranked.job
        score = ranked.score
        output.append({
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "is_remote": job.is_remote,
            "employment_type": job.employment_type,
            "url": job.url,
            "posted_date": str(job.posted_date) if job.posted_date else None,
            "salary_min": job.salary_min,
            "salary_max": job.salary_max,
            "salary_currency": job.salary_currency,
            "description": job.description,
            "skills_match": score.skills_match,
            "seniority_alignment": score.seniority_alignment,
            "location_fit": score.location_fit,
            "salary_fit": score.salary_fit,
            "company_score": ranked.company_score,
            "final_score": round(ranked.final_score, 4),
            "reasoning": score.reasoning,
            "config_key": config_key,
        })

    RANKED_JOBS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Scored and saved {len(output)} ranked jobs -> {RANKED_JOBS_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reweight-only",
        action="store_true",
        help="Recompute final_score from stored dimension scores without LLM calls",
    )
    args = parser.parse_args()

    if args.reweight_only:
        reweight_only()
    else:
        fresh_score()
