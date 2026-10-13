from pydantic import BaseModel, Field
from typing import Optional


class WorkExperience(BaseModel):
    company: str
    title: str
    duration: str
    summary: str


class Education(BaseModel):
    degree: str
    institution: str
    year: Optional[int] = None


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str]


class Job(BaseModel):
    title: str
    company: str
    location: str
    is_remote: bool
    description: str
    employment_type: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    url: str
    posted_date: Optional[str] = None


class CandidateProfile(BaseModel):
    name: str
    current_title: str
    years_of_experience: int
    skills: list[str] = Field(
        description="All technical skills including programming languages (Python, SQL, etc.), frameworks, and tools"
    )
    experience: list[WorkExperience]
    education: list[Education]
    projects: list[Project]
    inferred_roles: list[str] = Field(
        description="3-5 job titles that best match the candidate's background and skills"
    )
    open_to_remote: bool = Field(
        description="Infer from CV; default to True if not mentioned"
    )
    preferred_locations: list[str] = Field(
        description="Cities, regions, or countries mentioned in CV; empty list if none mentioned"
    )


class JobScore(BaseModel):
    skills_match: int = Field(ge=1, le=10, description="How many required/preferred skills does the candidate have? 10=perfect, 5=average, 1=poor")
    seniority_alignment: int = Field(ge=1, le=10, description="How well does the role seniority match the candidate's years of experience? 10=perfect fit")
    location_fit: int = Field(ge=1, le=10, description="How well does job location/remote policy match candidate preferences? 10=perfect")
    salary_fit: Optional[int] = Field(default=None, ge=1, le=10, description="Salary fit vs candidate expectations; omit entirely if salary not mentioned in the posting")
    reasoning: str = Field(description="One sentence naming the strongest signal driving these scores")


class RankedJob(BaseModel):
    job: Job
    score: JobScore
    company_score: int
    final_score: float
