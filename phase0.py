import os
from dotenv import load_dotenv
from openai import OpenAI
from opik.integrations.openai import track_openai
import opik

load_dotenv()

client = track_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


@opik.track(project_name=os.getenv("OPIK_PROJECT_NAME", "job-search-agent"))
def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = ask_llm("What does an AI Engineer do?")
    print(result)
