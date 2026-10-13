import os
from pathlib import Path
from dotenv import load_dotenv
import pypdf
from openai import OpenAI
from opik.integrations.openai import track_openai
import opik

from models import CandidateProfile

load_dotenv()

client = track_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

SYSTEM_PROMPT = """You are a CV parser. Extract structured information accurately from the CV text.
For skills, include all technical skills: programming languages (Python, SQL, R, etc.), frameworks, libraries, and tools. 
Do not create a separate languages field.
For projects, extract any personal or academic projects mentioned. If none are present, return an empty list.
For inferred_roles, suggest 3-5 job titles that best match the candidate's background and skills.
For open_to_remote, infer from any mentions of remote work preference, defaulting to True if not mentioned.
For preferred_locations, extract any cities, regions, or countries mentioned in the CV; return an empty list if none found."""

# Override inferred_roles with your own target roles, or leave empty to use LLM inference.
PREFERRED_ROLES: list[str] = []

# Flat list of cities and countries the agent will search. Override here or leave empty to use LLM inference.
SEARCH_LOCATIONS: list[str] = [
    "Delhi NCR", "Bangalore", "Mumbai", "Hyderabad", "Pune", "Chennai",
    "Dubai", "Singapore", "Malaysia", "Thailand",
    "Remote",
]


def extract_pdf_text(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def parse_cv(pdf_path: str) -> CandidateProfile:
    text = extract_pdf_text(pdf_path)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this CV:\n\n{text}"},
        ],
        response_format=CandidateProfile,
    )
    profile = response.choices[0].message.parsed
    if profile is None:
        raise ValueError("LLM refused to parse the CV. Check the input content.")
    return profile


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "resume" / "cv.pdf"
    profile = parse_cv(str(pdf_path))

    search_roles = PREFERRED_ROLES if PREFERRED_ROLES else profile.inferred_roles
    search_locations = SEARCH_LOCATIONS if SEARCH_LOCATIONS else profile.preferred_locations

    print(profile.model_dump_json(indent=2))
    print(f"\nRoles     : {search_roles}")
    print(f"Locations : {search_locations}")
