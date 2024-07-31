# Introduction

A quick sketch of the AI-candidate-screener idea.

Main code is in `main.py` and it just concatenates text about:

- the candidate
- the company
- the job

...into a basic prompt, and gets a reponse.

A starting point for further enchancements.


# Get started

```
conda env create -f environment.yml
# conda env update --file environment.yml --name culture_fit_bot
conda activate culture_fit_bot
# conda env remove --name culture_fit_bot
```

# Notes

- don't suggest keywords/jargon into CV, but rather tailor existing CV's keywords to the job description

methodology:
- from JD, list all the keywords
- from your CV, modify job titles to fit the role on the JD
- CV hard skills section: list all relevant keywords here
- what is the recruiter/hirer most interested in? Pay attention to the skills listed first or most often in the JD. Add it prominently in your CV, and mention it 2/3 times in the CV.


# CV examples

- mine
- https://sites.google.com/view/fellycikaya/about-me/my-cv
- https://theone9807.github.io/

# DB-data-driven CV

Started here: https://chatgpt.com/share/767e177f-9cea-43e8-ac58-21ee55d57ec2
