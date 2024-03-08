import os
import pathlib

import anthropic
import tiktoken
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# get cwd using pathlib
cwd = pathlib.Path(__file__).parent.absolute()
# cwd / ".." / "data/raw/company/about.txt"


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


def predict_claude(prompt: str) -> str:
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="You are a recruiter tasked with establishing whether a candidate is a good fit for a role at a given company.",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )
    return message.content


def _load_files_from_dictionary(data: dict) -> dict:
    data_copy = data.copy()
    for key, file_name in data.items():
        abs_file = cwd / ".." / file_name
        with open(abs_file, "r") as file:
            data_copy[key] = file.read().strip()
    return data_copy


def _count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding)
    encoded = enc.encode(text)
    return len(encoded)


def _load_entity(data: dict) -> str:

    # load each file
    loaded_data = _load_files_from_dictionary(data)

    # concatenate all the text
    loaded_text = " ".join(loaded_data.values())

    return loaded_text


def load_company():
    # load the company data
    company_data = {
        "about": "data/raw/company/about.txt",
        "homepage": "data/raw/company/homepage.txt",
        "leadership": "data/raw/company/leadership.txt",
        "working_here": "data/raw/company/working_here.txt",
    }

    return _load_entity(company_data)


def load_job():
    # load the job data
    job_data = {
        "job_description": "data/raw/company/job_description.txt",
    }

    return _load_entity(job_data)


def load_candidate():
    # load the candidate data
    candidate_data = {
        "cv": "data/raw/candidate/cv.txt",
        "cover_letter": "data/raw/candidate/cover_letter.txt",
    }

    return _load_entity(candidate_data)


"""
overemphasis on keywords (31.2%)
inaccurate interpretation of soft skills (26.3%)
inability to capture candidate potential (15.5%)
over-reliance on historical data (15.5%)

Modify the prompt to take care of the above issues.
"""

PROMPT_TEMPLATE = """The candidate wants to apply for this job with this company. Please establish whether the following candidate and company are a good fit for each other, and whether the candidate is a good fit for the job.

<company>
{company_text}
</company>

<job_description>
{job_description_text}
</job_description>

<candidate>
{candidate_text}
</candidate>
"""

# education, skills, experience, culture fit, work ethic, problem-solving abilities, leadership potential, adaptability, communication skills, teamwork, innovation & creativity, emotional intelligence
attributes = [
    "education",
    "skills",
    "experience",
    "culture fit",
    "work ethic",
    "problem-solving abilities",
    "leadership potential",
    "adaptability",
    "communication skills",
    "teamwork",
    "innovation & creativity",
    "emotional intelligence",
]


def main():
    company_text = load_company()
    company_token_len = _count_tokens(company_text)
    print(f"Company token length: {company_token_len}")

    candidate_text = load_candidate()
    candidate_token_len = _count_tokens(candidate_text)
    print(f"Candidate token length: {candidate_token_len}")

    job_text = load_job()
    job_token_len = _count_tokens(job_text)
    print(f"Job token length: {job_token_len}")

    prompt = PROMPT_TEMPLATE.format(
        company_text=company_text,
        candidate_text=candidate_text,
        job_description_text=job_text,
    )
    prompt_token_len = _count_tokens(prompt)
    print(f"Prompt token length: {prompt_token_len}")

    response = predict_claude(prompt)
    return response


response = main()
print(len(response))
print(response[0].text)

"""
Based on the information provided, there seems to be a good fit between the candidate, Juan Uys, and the company, Climate Policy Radar, for the Machine Learning Engineer role. Here's why:

1. Alignment of mission and values:
- Juan expresses a strong passion for addressing climate change and saving the planet, which aligns with Climate Policy Radar's mission to accelerate the transition to a low-carbon, resilient, and just world.
- Juan's personal actions, such as driving an EV, renovating his home for energy efficiency, and instilling planet-saving values in his children, demonstrate his commitment to the cause.

2. Relevant technical skills and experience:
- The job description requires expertise in Machine Learning, AI, LLM, AWS, NLP, and end-to-end software development, all of which Juan possesses.
- Juan has been working on document understanding and data extraction projects, including his side project PDFCrunch, which is directly relevant to Climate Policy Radar's work in analyzing climate law and policy documents.
- He has experience with Python and ML technologies, as well as research into various AI models and techniques, which can contribute to the development of Climate Policy Radar's data extraction tool.

3. Proven track record and leadership:
- With over 20 years of experience, Juan is likely to be a reliable and skilled contributor to the project.
- He has demonstrated leadership skills by growing and leading teams in his previous roles, which can be valuable for shaping the strategy and execution of the project.

4. Remote work experience and self-direction:
- As a contractor who has mostly worked remotely since 2014, Juan is well-suited for the remote nature of the role and has proven his ability to work independently and be self-directed.

5. Startup experience and understanding of business needs:
- Having been a co-founder and CTO of a startup, Juan understands the needs and expectations of a business from a service provider's perspective, which can help him effectively contribute to Climate Policy Radar's goals.

In summary, Juan Uys appears to be a strong candidate for the Machine Learning Engineer role at Climate Policy Radar, given his relevant technical skills, experience, passion for the cause, and alignment with the company's mission and values.
"""
