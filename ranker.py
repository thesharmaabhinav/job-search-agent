import json
import math
import os
from datetime import date
from pathlib import Path

import opik
from dotenv import load_dotenv
from openai import OpenAI
from opik.integrations.openai import track_openai

from models import CandidateProfile, Job, JobScore, RankedJob

load_dotenv()

COMPANY_CACHE_PATH = Path(__file__).parent / "company_cache.json"
_RANKER_CONFIG_PATH = Path(__file__).parent / "ranker_config.json"

_RANKER_CONFIG_DEFAULTS: dict = {
    "w_skills":       0.75,
    "w_seniority":    0.00,
    "w_location":     0.25,
    "w_company":      0.00,
    "w_salary_bonus": 0.00,
    "top_k":            15,
    "embedding_model": "text-embedding-3-small",
    "llm_model":       "gpt-4o-mini",
    "judge_prompt": (
        "Score job-candidate fit on a 1-10 scale per dimension. "
        "Be strict: 10=perfect, 5=average, 1=poor. "
        "Omit salary_fit if salary is not listed in the posting."
    ),
}


def _load_ranker_config() -> dict:
    if _RANKER_CONFIG_PATH.exists():
        return {**_RANKER_CONFIG_DEFAULTS, **json.loads(_RANKER_CONFIG_PATH.read_text())}
    return _RANKER_CONFIG_DEFAULTS.copy()


RANKER_CONFIG: dict = _load_ranker_config()

client = track_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def _load_company_cache() -> dict:
    if COMPANY_CACHE_PATH.exists():
        return json.loads(COMPANY_CACHE_PATH.read_text())
    return {}


def _save_company_cache(cache: dict) -> None:
    COMPANY_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _build_candidate_text(profile: CandidateProfile) -> str:
    return (
        f"Target roles: {', '.join(profile.inferred_roles)}. "
        f"Skills: {', '.join(profile.skills)}. "
        f"Experience titles: {', '.join(exp.title for exp in profile.experience)}. "
        f"Years of experience: {profile.years_of_experience}."
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def embedding_prefilter(jobs: list[Job], profile: CandidateProfile, top_k: int | None = None) -> list[Job]:
    top_k = top_k if top_k is not None else RANKER_CONFIG["top_k"]
    candidate_text = _build_candidate_text(profile)
    job_texts = [f"{job.title} {job.description[:1500]}" for job in jobs]

    # Single batch call — one token cost for all embeddings
    response = client.embeddings.create(
        model=RANKER_CONFIG["embedding_model"],
        input=[candidate_text] + job_texts,
    )

    candidate_emb = response.data[0].embedding
    scored = [
        (_cosine_similarity(candidate_emb, item.embedding), job)
        for item, job in zip(response.data[1:], jobs)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [job for _, job in scored[:top_k]]


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def _get_company_score(company: str, cache: dict) -> int:
    if company in cache:
        return cache[company]["score"]

    response = client.chat.completions.create(
        model=RANKER_CONFIG["llm_model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Score the given company 1-10 for an AI/data/software engineering candidate. "
                    "Consider reputation, tech industry relevance, and stability. "
                    'Reply with JSON only: {"score": <int 1-10>, "reasoning": "<one sentence>"}'
                ),
            },
            {"role": "user", "content": company},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    cache[company] = {
        "score": result["score"],
        "reasoning": result["reasoning"],
        "cached_at": date.today().isoformat(),
    }
    return result["score"]


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def _score_job(job: Job, profile: CandidateProfile) -> JobScore:
    salary_context = (
        f"Salary: {job.salary_currency} {job.salary_min}-{job.salary_max}."
        if job.salary_min
        else "Salary not listed."
    )
    locations_str = ", ".join(profile.preferred_locations) or "flexible"

    return client.beta.chat.completions.parse(
        model=RANKER_CONFIG["llm_model"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": RANKER_CONFIG["judge_prompt"],
            },
            {
                "role": "user",
                "content": (
                    f"Candidate: {profile.years_of_experience} yrs exp, "
                    f"title={profile.current_title}, "
                    f"skills=[{', '.join(profile.skills)}], "
                    f"preferred locations=[{locations_str}], "
                    f"open to remote={profile.open_to_remote}\n\n"
                    f"Job: {job.title} at {job.company}, "
                    f"location={job.location}, remote={job.is_remote}, {salary_context}\n"
                    f"Description: {job.description[:1200]}"
                ),
            },
        ],
        response_format=JobScore,
    ).choices[0].message.parsed


def _compute_final_score(score: JobScore, company_score: int) -> float:
    base = (
        (score.skills_match / 10)        * RANKER_CONFIG["w_skills"]
        + (score.seniority_alignment / 10) * RANKER_CONFIG["w_seniority"]
        + (score.location_fit / 10)        * RANKER_CONFIG["w_location"]
        + (company_score / 10)             * RANKER_CONFIG["w_company"]
    )
    salary_bonus = (score.salary_fit / 10) * RANKER_CONFIG["w_salary_bonus"] if score.salary_fit is not None else 0.0
    return base + salary_bonus


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def score_jobs_for_eval(jobs: list[Job], profile: CandidateProfile) -> list[RankedJob]:
    """Score all jobs with the LLM judge, bypassing the embedding pre-filter. For eval use only."""
    company_cache = _load_company_cache()
    ranked: list[RankedJob] = []
    for job in jobs:
        company_score = _get_company_score(job.company, company_cache)
        job_score = _score_job(job, profile)
        ranked.append(RankedJob(
            job=job,
            score=job_score,
            company_score=company_score,
            final_score=_compute_final_score(job_score, company_score),
        ))
    _save_company_cache(company_cache)
    return sorted(ranked, key=lambda x: x.final_score, reverse=True)


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def rank_jobs(jobs: list[Job], profile: CandidateProfile) -> list[RankedJob]:
    shortlisted = embedding_prefilter(jobs, profile)

    company_cache = _load_company_cache()
    ranked: list[RankedJob] = []

    for job in shortlisted:
        company_score = _get_company_score(job.company, company_cache)
        job_score = _score_job(job, profile)
        ranked.append(RankedJob(
            job=job,
            score=job_score,
            company_score=company_score,
            final_score=_compute_final_score(job_score, company_score),
        ))

    _save_company_cache(company_cache)
    return sorted(ranked, key=lambda x: x.final_score, reverse=True)
