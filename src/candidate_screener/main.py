import re
from pathlib import Path

import anthropic
import tiktoken
from dotenv import find_dotenv, load_dotenv

from candidate_screener.model import Candidate

load_dotenv(find_dotenv())

# get cwd using pathlib
cwd = Path(__file__).parent.absolute()
# print(cwd)
# cwd / ".." / "data/raw/company1/about.txt"


client = anthropic.Anthropic()


def predict_claude(prompt: str, system_prompt: str) -> str:
    """Given a prompt, predict a completion.

    max_tokens: is 4096 on the downstream backend. See https://docs.anthropic.com/claude/docs/models-overview#model-comparison

    Args:
        prompt (str): the prompt/instruction
        system_prompt (str): the role of the model to assume

    Returns:
        str: the prediction
    """
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        # top_p=0.75,
        # top_k=30,
    )
    return message.content


def _load_files_from_dictionary(data: dict) -> dict:
    data_copy = data.copy()
    for key, file_name in data.items():
        abs_file = cwd / ".." / ".." / file_name
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
    loaded_text = "\n\n".join(loaded_data.values())

    return loaded_text


def load_company(company_name: str = "company1"):
    # load the company data
    company_data = {
        "about": f"data/raw/{company_name}/about.txt",
        "homepage": f"data/raw/{company_name}/homepage.txt",
        "leadership": f"data/raw/{company_name}/leadership.txt",
        "working_here": f"data/raw/{company_name}/working_here.txt",
    }

    return _load_entity(company_data)


def load_job(company_name: str = "company1"):
    # load the job data
    job_data = {
        "job_description": f"data/raw/{company_name}/job_description.txt",
    }

    return _load_entity(job_data)


def load_candidate(candidate_name: str = "candidate1") -> Candidate:
    # load the candidate data
    candidate_data = {
        "cv": f"data/raw/{candidate_name}/cv.txt",
        "cover_letter": f"data/raw/{candidate_name}/cover_letter.txt",
    }

    candidate_raw_data = _load_files_from_dictionary(candidate_data)

    candidate_data = Candidate(**candidate_raw_data)

    return candidate_data


# load_candidate("candidate1")

"""
overemphasis on keywords (31.2%)
inaccurate interpretation of soft skills (26.3%)
inability to capture candidate potential (15.5%)
over-reliance on historical data (15.5%)

Modify the prompt to take care of the above issues.
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

attributes_extra = [
    {
        "name": "education",
        "prompt": "If the company doesn't require any specific educational requirements, then the candidate's education is less important. However, if the company has a specific requirement, like a degree or diploma, the candidate's education is more important, and a suitable match ranking need to be predicted.",
    },
    {
        "name": "skills",
        "prompt": "The candidate's skills are important to match with the company's requirements. If the candidate has the exact required skills, the harmony can be scored 1.0, and the conflict score can be very low or 0.0. If a candidate has adjacent skills that are quite similar to the required skills, the match can be ranked lower than 1.0, but still highly. If the candidate has no relevant skills, the match can be ranked lower.",
    },
    {
        "name": "experience",
        "prompt": "Pay special attention to whether the company explicitly requires experience in any specific industries, technologies, or tools, and rank accordingly. However, if the company is not specific about the required experience, and the candidate doesn't have a lot of experience, then other factors can be taken into account to support a candidate's experience, like the candidate's side-projects, hobbies, or similar activities.",
    },
    {
        "name": "culture fit",
        # 0.8
        # "prompt": "Pay special attention to culture fit, as a candidate might have viewpoints or hobbies which are at odds with the company's culture, mission, and values.",
        # 0.85 (worse than 0.8 - we want to see ~0.5)
        "prompt": "Make sure to explicitly consider whether the candidate's stated hobbies, interests or other personal details potentially conflict with the core mission, values and focus areas of the company. Evaluate both alignment and potential misalignment.",
    },
    {
        "name": "work ethic",
        "prompt": None,
    },
    {
        "name": "problem-solving abilities",
        "prompt": None,
    },
    {
        "name": "leadership potential",
        "prompt": None,
    },
    {
        "name": "adaptability",
        "prompt": None,
    },
    {
        "name": "communication skills",
        "prompt": None,
    },
    {
        "name": "teamwork",
        "prompt": None,
    },
    {
        "name": "innovation & creativity",
        "prompt": None,
    },
    {
        "name": "emotional intelligence",
        "prompt": None,
    },
]


def natural_join(words):
    return ", ".join(words[:-1]) + " and " + words[-1]


PROMPT_TEMPLATE = """Please establish whether the following candidate and company are a good fit for each other, and whether the candidate is a good fit for the job.

The company's data is inside the <company></company> tag, and the job that the company is hiring for is inside the <job_description></job_description> tag.

The candidate's data is split into CV inside the <candidate_cv></candidate_cv> tag, cover letter inside the <candidate_cover_letter></candidate_cover_letter> tag, and extra text inside the <candidate_extra_text></candidate_extra_text> tag. The extra text can include any additional information about the candidate that might be relevant, and that isn't included in the CV or cover letter, like the text from the candidate's personal website's About page.

Match the candidate on each of the following attributes, with separate scores between 0 and 1 for "harmony" and "conflict": {attributes_to_match_on}.{attributes_extra_text}

Please give detailed reasons for your scores for each of the attributes. Note that harmony and conflict isn't mutually exclusive, and a candidate can have both high harmony and conflict for the same attribute. For instance, a candidate might have a high harmony score for culture fit, because they're educated on a postgraduate level just like the company founders, but they might also have a high conflict score for culture fit, because they partake in a hobby which might be at odds with the company's mission and values, like hunting animals whereas the company operates in animal welfare.

Use YAML formatting and wrap with a <match></match> tag, e.g.:
<match>
education:
  harmony:
    score: value between 0 and 1
    reason: "A thoughtful reason explaining why this candidate got this score for being aligned with the company's educational requirements."
  conflict:
    score: value between 0 and 1
    reason: "A thoughtful reason explaining why this candidate got this score for being misaligned with the company's educational requirements."
"communication skills":
  harmony:
    score: value between 0 and 1
    reason: "A thoughtful reason explaining why this candidate got this score for being aligned with the company."
  conflict:
    score: value between 0 and 1
    reason: "A thoughtful reason explaining why this candidate got this score for being misaligned with the company."
further_attributes:
  harmony:
    score: value between 0 and 1
    reason: "A thoughtful reason explaining why this candidate got this score for being aligned with the company."
  conflict:
    score: value between 0 and 1
    reason: "A thoughtful reason explaining why this candidate got this score for being misaligned with the company."
</match>

And then finally, produce a summary, wrapped in a <summary></summary> tag, of the candidate's overall suitability for the job, and their overall suitability for the company.

Here is the data:

<company>
{company_text}
</company>

<job_description>
{job_description_text}
</job_description>

<candidate_cv>
{candidate_cv_text}
</candidate_cv>
<candidate_cover_letter>
{candidate_cover_letter_text}
</candidate_cover_letter>
<candidate_extra_text>
{candidate_extra_text}
</candidate_extra_text>

Note: discrimination is illegal.
"""


def remove_tags(tag: str, text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"(?i)<\/?" + tag + ">", "", text)


# remove_tags("company", "This is a <Company>test</compaNy> string with <COMPANY>multiple</company> instances.")


def get_prompt(
    company_name: str = "company1", candidate_name: str = "candidate1"
) -> str:
    company_text = load_company(company_name)
    company_text = remove_tags("company", company_text)
    company_token_len = _count_tokens(company_text)
    print(f"Company token length: {company_token_len}")

    job_text = load_job(company_name)
    job_text = remove_tags("job_description", job_text)
    job_token_len = _count_tokens(job_text)
    print(f"Job token length: {job_token_len}")

    candidate = load_candidate(candidate_name)
    candidate_token_len = _count_tokens(candidate.as_str())
    print(f"Candidate token length: {candidate_token_len}")
    candidate_cv = remove_tags("candidate_cv", candidate.cv)
    candidate_cover_letter = remove_tags(
        "candidate_cover_letter", candidate.cover_letter
    )
    candidate_extra = remove_tags("candidate_extra_text", candidate.extra)

    attributes_extra_text = "\n- ".join(
        [
            f"{attr['name']}: {attr['prompt']}"
            for attr in attributes_extra
            if attr["prompt"] is not None
        ]
    )
    if attributes_extra_text:
        attributes_extra_text = (
            "\nA few extra notes about some of the attributes:\n"
            + "- "
            + attributes_extra_text
        )

    prompt = PROMPT_TEMPLATE.format(
        company_text=company_text,
        candidate_cv_text=candidate_cv,
        candidate_cover_letter_text=candidate_cover_letter,
        candidate_extra_text=candidate_extra,
        job_description_text=job_text,
        attributes_to_match_on=natural_join(attributes),
        attributes_extra_text=attributes_extra_text,
    )
    return prompt


# print(get_prompt())


def get_match(
    company_name: str = "company1", candidate_name: str = "candidate1"
) -> tuple[str, str]:

    prompt = get_prompt(company_name, candidate_name)
    prompt_token_len = _count_tokens(prompt)
    print(f"Prompt token length: {prompt_token_len}")

    system_prompt = "You are an unbiased 3rd party who is tasked with establishing whether a candidate is a good fit for a role at a given company."
    response = predict_claude(prompt, system_prompt)
    return prompt, response


# company_name = "company1"
# candidate_name = "candidate3"
# prompt, response = get_match(company_name=company_name, candidate_name=candidate_name)
# print(len(response))
# print(prompt)
# print(response[0].text)
