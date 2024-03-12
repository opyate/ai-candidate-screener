import re
from pathlib import Path

import anthropic
import tiktoken
from dotenv import find_dotenv, load_dotenv

from .model import Candidate

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
        "prompt": "The candidate's skills are important to match with the company's requirements. If the candidate has the exact required skills, the match can be ranked 1.0. If a candidate has adjacent skills that are quite similar to the required skills, the match can be ranked lower than 1.0, but still highly. If the candidate has no relevant skills, the match can be ranked lower.",
    },
    {
        "name": "experience",
        "prompt": "Pay special attention to whether the company explicitly requires experience in any specific industries, technologies, or tools, and rank accordingly. However, if the company is not specific about the required experience, and the candidate doesn't have a lot of experience, then other factors can be taken into account to support a candidate's experience, like the candidate's side-projects, hobbies, or similar activities.",
    },
    {
        "name": "culture fit",
        "prompt": "Pay special attention to culture fit, as a candidate might have viewpoints or hobbies which are at odds with the company's culture, mission, and values.",
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

The candidate's data is split into CV as per the <candidate_cv></candidate_cv> tag, cover letter as per the <candidate_cover_letter></candidate_cover_letter> tag, and extra text as per the <candidate_extra_text></candidate_extra_text> tag. The extra text can include any additional information about the candidate that might be relevant, and that isn't included in the CV or cover letter, like the text from their website's About page.

Match the candidate on each of the following attributes: {attributes_to_match_on}.{attributes_extra_text}

Please give detailed reasons for your match scores for each of the attributes.

Use YAML formatting and wrap with a <match></match> tag, e.g.:
<match>
education:
  match: value between 0 and 1
  reason: "The candidate has a degree in X, which is good/bad, because the company wants Y"
"communication skills":
  match: value between 0 and 1
  reason: "Judging by the candidate's fluency with words in their cover letter, and their structured approach to their CV, they seem to have good communication skills."
further_attributes:
  match: value between 0 and 1
  reason: "etc"
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


company_name = "company1"
candidate_name = "candidate3"
prompt, response = get_match(company_name=company_name, candidate_name=candidate_name)
print(len(response))
print(response[0].text)

"""
education:
  match: 0.8
  reason: "The candidate has an AI diploma and has completed several relevant courses in deep learning, NLP, and data analysis. While not a traditional degree, this education aligns well with the technical requirements of the role."

skills:
  match: 0.9
  reason: "The candidate has strong skills in machine learning, AI, LLMs, AWS, and NLP, which are key requirements for the role. They also have experience with Python and developing proof of concepts and end-to-end software solutions."

experience:
  match: 0.9
  reason: "The candidate has highly relevant experience working on document understanding and data extraction projects, including recent work reducing risk quote processing time using ML technologies. This aligns very well with the job description."

"culture fit":
  match: 0.7
  reason: "The candidate's passion for machine learning and experience as an independent consultant suggest they would work well in a mission-driven startup environment. However, their personal interests in auto racing and modifications may not directly align with the company's focus on climate change solutions."

"work ethic":
  match: 0.8
  reason: "The candidate's long history of successful projects and leadership roles suggests a strong work ethic. However, more information would be needed to fully assess this."

"problem-solving abilities":
  match: 0.8
  reason: "The candidate's experience delivering end-to-end solutions and working on complex ML projects demonstrates solid problem-solving skills. Their background in mathematics further supports this."

"leadership potential":
  match: 0.8
  reason: "The candidate has held team lead roles and grown teams in the past, suggesting good leadership potential. They also have experience as a startup founder and CTO."

adaptability:
  match: 0.8
  reason: "The candidate's varied experience across industries and technologies shows an ability to adapt to different projects and environments."

"communication skills":
  match: 0.7
  reason: "While the CV doesn't give direct evidence of communication skills, the candidate's experience in leadership roles and as an independent consultant suggests an ability to communicate effectively. More information would help assess this further."

teamwork:
  match: 0.7
  reason: "The candidate has worked successfully in team environments in the past. However, their more recent focus on independent consulting work makes it harder to assess their current teamwork abilities."

"innovation & creativity":
  match: 0.8
  reason: "The candidate's experience with cutting-edge ML technologies and research suggests an innovative mindset. Their awards from hackathons and startup competitions also point to creativity."

"""

# 2024-03-12:
"""
Here is my assessment of the candidate's fit for the job and company:

<match>
education:
  match: 0.8
  reason: "The candidate has relevant education in AI and machine learning, including an AI diploma and NVidia certifications in data parallelism, multi-GPU computing, and transformer-based NLP. While not a strict requirement, this aligns well with the ML and NLP skills needed for the role."

skills:
  match: 0.9
  reason: "The candidate has highly relevant skills in machine learning, AI, LLMs, NLP, and AWS, which match the key requirements for the role. Experience with .Net is lacking but the job description notes other languages like Python are acceptable, which the candidate is proficient in."

experience:
  match: 0.95
  reason: "With 20+ years of experience, including 10+ years as an independent ML consultant working on document understanding and data extraction projects, the candidate brings highly relevant and extensive experience. Work on reducing risk quote processing time and structured data extraction align closely with the role. Experience delivering end-to-end software solutions is also evident."

"culture fit":
  match: 0.85
  reason: "The candidate's passion for machine learning and clear communication align with the company's commitment to excellence and truth-seeking in climate data and policy. Independent work and leadership experience fit the startup environment. No obvious misalignments with the company's collaborative, transparent and emotionally intelligent culture."

"work ethic":
  match: 0.9
  reason: "Extensive consulting experience, leadership roles, and awards suggest a strong work ethic. Maintaining an active career while pursuing additional education also demonstrates dedication and drive."

"problem-solving abilities":
  match: 0.9
  reason: "Tackling complex data extraction and document understanding problems as an independent consultant exhibits strong problem-solving skills. Expanding platforms for new territories and aggregating disparate data sources also showcase this ability."

"leadership potential":
  match: 0.85
  reason: "The candidate has taken on team leadership roles at multiple points in their career, growing teams and mentoring others. Startup CTO experience also demonstrates leadership capabilities. While the current role may not require leadership, the potential is clearly there."

adaptability:
  match: 0.9
  reason: "A career spanning multiple industries, technologies, and countries demonstrates adaptability. Pivoting from a long-term Java focus to Python and ML also shows the ability to adapt and learn new skills."

"communication skills":
  match: 0.8
  reason: "The candidate highlights clear communication to stakeholders as a strength. Leadership roles also require effective communication. The cover letter is brief but professional and to-the-point."

teamwork:
  match: 0.8
  reason: "Extensive team leadership experience and a collaborative history working with small teams and mentoring others. No direct mention of teamwork in the cover letter but no red flags either."

"innovation & creativity":
  match: 0.85
  reason: "Founding an ML-focused PDF data extraction startup demonstrates innovation. Creative problem-solving is evident in the candidate's projects and consulting work."

"emotional intelligence":
  match: 0.8
  reason: "Hard to assess directly, but no red flags in communication style or experience. Leadership roles and independent consulting require a degree of emotional intelligence to succeed."
</match>

<summary>
In summary, this candidate appears to be an excellent fit for both the Machine Learning Engineer role and the Climate Policy Radar company. They bring highly relevant skills and extensive experience in machine learning, NLP, data extraction, and delivering end-to-end solutions. The candidate's passion for ML and clear communication align well with the company's commitment to excellence and truth-seeking.

While direct experience with .Net is lacking, proficiency in Python and other key skills make the candidate well-suited for the role. The candidate also demonstrates leadership potential, adaptability, problem-solving abilities, and likely emotional intelligence that would enable them to thrive in CPR's mission-driven startup environment.

No major red flags in terms of culture fit or work ethic. The candidate's education, while not a strict requirement, is a relevant asset. Overall, I would highly recommend considering this candidate for the Machine Learning Engineer position.
</summary>
"""
