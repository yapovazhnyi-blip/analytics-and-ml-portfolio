# Module 4 — Content Discovery Analysis Memo

A Netflix-style DS investigation notebook demonstrating senior data scientist reasoning. The goal is not the model — it is framing the right question.

## Run

```bash
python module4_analysis_memo/analysis_memo.py
```

Or convert to a Jupyter notebook:

```bash
pip install jupytext
jupytext --to notebook analysis_memo.py
jupyter notebook analysis_memo.ipynb
```

## Structure

```
1. Problem framing     why does discovery matter for retention?
2. Metric definition   what exactly is "discovery success"?
3. Data overview       sanity checks, arm balance
4. Primary analysis    CTR + long-tail engagement by arm
5. OLS regression      HC3-robust SEs, activity controls
6. Cohort breakdown    power users vs casual vs lurkers
7. Findings            plain-language with uncertainty bounds
8. What to measure next  30-day retention, genre diversity, novelty decay
```

## The discovery metric

Standard A/B tests measure CTR. This memo argues for a better primary metric: **long-tail engagement rate** — the fraction of clicks on movies ranked >500 by popularity. A user clicking on a 2003 Estonian drama is discovering content. A user clicking on a blockbuster is not.

## What to measure next

The "what to measure next" section is the most important part — it demonstrates senior DS thinking:
- 30-day retention by cohort (leading → lagging metric relationship)
- Genre diversity per user (Gini coefficient on genre distribution)
- Novelty decay at 90 days (does the lift persist?)
- Cold-start gap (Two-Tower fails for new users — test a hybrid)
