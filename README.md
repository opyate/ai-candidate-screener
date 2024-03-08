# Introduction

Just had an idea: culture fit.

A company wants to hire for a new role. But they'd get hundreds of CVs and cover letters.

Have a system which can analyse the company's website (especially the culture bits: about us, the way we work, our values, etc).

Then, for each applicant, build up a comprehensive profile of the applicant:
- analyse their CV
- analyse their cover letter
- analyse the content at any of the URLs in their CV/cover letter (e.g. it might point to their personal/hobby websites)

Then rank the matches, and write a report about the match/fit.


# Get started

```
conda env create -f environment.yml
# conda env update --file environment.yml --name culture_fit_bot
conda activate culture_fit_bot
# conda env remove --name culture_fit_bot
```