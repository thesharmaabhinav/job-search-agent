import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import opik
from _opik import ScoreResult
from dotenv import load_dotenv
from opik.evaluation.test_result import TestResult

from cv_parser import parse_cv
from models import Job
from ranker import RANKER_CONFIG, score_jobs_for_eval

load_dotenv()

# ── Update both before each eval run ─────────────────────────────────────────
EXPERIMENT_NAME  = "Final_high_skill_no_seniority"
EXPERIMENT_NOTES = "w_company=0.00, w_skills=0.75, w_seniority=0.00, w_location=0.25"
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_PATH      = Path(__file__).parent / "jobs_snapshot.json"
LABELS_PATH        = Path(__file__).parent / "labels.json"
JOB_SCORES_CACHE_PATH = Path(__file__).parent / "job_scores_cache.json"
DATASET_NAME       = "job-search-eval"
PROJECT_NAME       = os.getenv("OPIK_PROJECT_NAME", "job-search-agent")


def _config_key() -> str:
    """Short hash of the judge prompt + model. Changes only when LLM behaviour changes."""
    sig = RANKER_CONFIG["judge_prompt"] + RANKER_CONFIG["llm_model"]
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def _load_scores_cache() -> dict:
    if JOB_SCORES_CACHE_PATH.exists():
        return json.loads(JOB_SCORES_CACHE_PATH.read_text())
    return {}


def _save_scores_cache(cache: dict) -> None:
    JOB_SCORES_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _final_score_from_cache(entry: dict) -> float:
    """Recompute final_score from cached dimension scores using current RANKER_CONFIG weights."""
    base = (
        (entry["skills_match"] / 10)        * RANKER_CONFIG["w_skills"]
        + (entry["seniority_alignment"] / 10) * RANKER_CONFIG["w_seniority"]
        + (entry["location_fit"] / 10)        * RANKER_CONFIG["w_location"]
        + (entry["company_score"] / 10)        * RANKER_CONFIG["w_company"]
    )
    sf = entry.get("salary_fit")
    salary_bonus = (sf / 10) * RANKER_CONFIG["w_salary_bonus"] if sf is not None else 0.0
    return base + salary_bonus


def _precision_at_10(results: list[TestResult]) -> ScoreResult:
    sorted_r = sorted(results, key=lambda r: r.test_case.task_output["final_score"], reverse=True)
    top_10 = sorted_r[:10]
    relevant = sum(1 for r in top_10 if r.test_case.task_output["reference_relevance"] > 0)
    return ScoreResult(name="precision_at_10", value=relevant / min(10, len(top_10)))


def _ndcg_at_10(results: list[TestResult]) -> ScoreResult:
    sorted_r = sorted(results, key=lambda r: r.test_case.task_output["final_score"], reverse=True)
    top_10 = sorted_r[:10]
    dcg = sum(
        r.test_case.task_output["reference_relevance"] / math.log2(i + 2)
        for i, r in enumerate(top_10)
    )
    all_rels = sorted(
        [r.test_case.task_output["reference_relevance"] for r in results],
        reverse=True,
    )
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(all_rels[:10]))
    return ScoreResult(name="ndcg_at_10", value=dcg / idcg if idcg > 0 else 0.0)


def main(experiment_name: str | None = None) -> None:
    name = experiment_name or EXPERIMENT_NAME
    if not SNAPSHOT_PATH.exists():
        print("No snapshot found. Run: uv run python agent.py --snapshot")
        return
    if not LABELS_PATH.exists():
        print("No labels found. Run: uv run python label_jobs.py --import")
        return

    jobs_raw: list[dict] = json.loads(SNAPSHOT_PATH.read_text())
    labels: dict[str, int] = json.loads(LABELS_PATH.read_text())
    jobs = [Job(**j) for j in jobs_raw if j["url"] in labels]
    print(f"{len(jobs)} labelled jobs loaded.")

    if len(jobs) < 10:
        print("Warning: fewer than 10 labelled jobs -- metrics will be unreliable.")

    pdf_path = Path(__file__).parent / "resume" / "cv.pdf"
    profile = parse_cv(str(pdf_path))

    # ── Score jobs (LLM calls only for uncached entries) ─────────────────────
    scores_cache = _load_scores_cache()
    config_key   = _config_key()

    uncached = [j for j in jobs if f"{config_key}:{j.url}" not in scores_cache]
    cached_count = len(jobs) - len(uncached)

    if uncached:
        print(f"Scoring {len(uncached)} jobs via LLM judge ({cached_count} from cache)...")
        newly_scored = score_jobs_for_eval(uncached, profile)
        for r in newly_scored:
            scores_cache[f"{config_key}:{r.job.url}"] = {
                "skills_match":        r.score.skills_match,
                "seniority_alignment": r.score.seniority_alignment,
                "location_fit":        r.score.location_fit,
                "salary_fit":          r.score.salary_fit,
                "company_score":       r.company_score,
            }
        _save_scores_cache(scores_cache)
    else:
        print(f"All {len(jobs)} jobs loaded from scores cache — no LLM calls needed.")

    score_lookup = {
        j.url: {
            **scores_cache[f"{config_key}:{j.url}"],
            "final_score": _final_score_from_cache(scores_cache[f"{config_key}:{j.url}"]),
        }
        for j in jobs
    }
    # ─────────────────────────────────────────────────────────────────────────

    # Clear + repopulate dataset so stale items from previous runs don't pollute metrics
    opik_client = opik.Opik()
    dataset = opik_client.get_or_create_dataset(name=DATASET_NAME, project_name=PROJECT_NAME)
    dataset.clear()
    dataset.insert([
        {"input": j.model_dump(), "reference_relevance": labels[j.url]}
        for j in jobs
    ])

    def task(item: dict) -> dict:
        cached = score_lookup[item["input"]["url"]]
        return {**cached, "reference_relevance": item["reference_relevance"]}

    result = opik.evaluate(
        experiment_name=name,
        dataset=dataset,
        task=task,
        experiment_config={**RANKER_CONFIG, "notes": EXPERIMENT_NOTES},
        experiment_scoring_functions=[_precision_at_10, _ndcg_at_10],
        project_name=PROJECT_NAME,
        task_threads=1,
    )

    print("\nEval complete.")
    for score in result.experiment_scores:
        print(f"  {score.name:<16} {score.value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, help="Opik experiment name (overrides EXPERIMENT_NAME)")
    args = parser.parse_args()
    main(experiment_name=args.experiment_name)
