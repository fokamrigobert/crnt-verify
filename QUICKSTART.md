# QUICKSTART — what these files are and how to run them

## The one-paragraph version

You built a **grader** (Phase 1) and then wrapped it in an **exam system**
(Phase 2). Phase 1 answers *"is this answer correct?"*. Phase 2 answers
*"generate a fresh problem, ask a model, score it, repeat."* You need
Phase 1 for Phase 2 to mean anything; you need Phase 2 for anyone to
train or benchmark against your work.

## The analogy that makes it click

You ran an engineering school for twelve years, so:

| | Role | Files |
|---|---|---|
| **Phase 1** | The **examiner**. `crnt_solver.py` is the professor who works out the model answers. `crnt_checker.py` is the marking scheme — the one that knows a student who wrote the conservation law scaled by 2 is still *right*. | `crnt_solver.py`, `crnt_checker.py` |
| **Phase 2** | The **examination hall**. Sets a fresh paper for every candidate, collects the script, hands it to the examiner, records the mark, calls the next candidate. | `crnt_gym.py`, `crnt_verifiers_env.py` |

An AI lab training a model needs the exam hall, not just the marking
scheme — because training means sitting the exam millions of times.

## Install

```bash
git clone https://github.com/fokamrigobert/crnt-verify
cd crnt-verify
pip install -r requirements.txt          # just sympy — Phase 1 + the environment core
```

Optional, only for the two extras:

```bash
pip install anthropic                    # to run evaluate.py against a real model
pip install -r requirements-env.txt      # to publish to Prime Intellect's Hub
```

## Run these in this order

### 1. See it work — no API key, no setup

```bash
python3 demo.py
```

Walks through one problem slowly and prints every stage: the generated
network, the exact prompt a model would see, the answer key, and what
happens when you submit a correct answer, a *differently-written* correct
answer, a wrong answer, and unparseable rambling. **Start here.** If you
read only one thing, read this output.

### 2. Check the environment itself is correct

```bash
python3 crnt_gym.py
```

Runs four fake policies over 15 problems each and asserts the results:

```
Oracle policy (exact ground truth):          mean reward = 1.000
Rewritten-but-correct policy (scaled law):   mean reward = 1.000   <- the important one
Broken policy (wrong coefficients):          mean reward = 0.000
Garbage / unparseable policy:                mean reward = 0.000
```

If that second line ever drops below 1.000, the whole premise of the
project has broken — it's the regression test that matters most.

### 3. Score a real model

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 evaluate.py --n 20 --difficulty 3
```

Prints per-problem pass/fail, then an accuracy figure, and writes every
transcript to `eval_results.json`. Costs a few cents for 20 problems.

## "How do I know the environment performs well?"

This is two different questions, and conflating them is the usual trap:

**Is the environment correct?** → `python3 crnt_gym.py`. This is
pass/fail, not a percentage. The environment doesn't have an accuracy any
more than a ruler has a length-it-measures — it's either right or broken.

**How hard is it / how good is a model on it?** → `python3 evaluate.py`.
*This* produces a percentage, and that percentage is a property of the
**model**, not of the environment. Read it like this:

| Model accuracy | What it means | What to do |
|---|---|---|
| ~100% | Too easy. No training signal — every rollout gets reward 1. | Raise `--difficulty` |
| 20–80% | The useful band. RL gets traction here. | Good, use it |
| ~0% | Too hard **or** something is broken | Open `eval_results.json` and read actual model outputs before concluding anything |

That last row matters. A formatting mismatch and genuine incapacity look
*identical* from the accuracy number alone. Always read a few raw
transcripts before reporting a headline figure — which is, in miniature,
exactly the research discipline you already apply to your own numerical
claims.

## Where the files go

All of them, flat, in the repo root — no subfolders:

```
crnt-verify/
├── crnt_solver.py           Phase 1 — computes the answer key
├── crnt_checker.py          Phase 1 — grades an answer (the core contribution)
├── crnt_gym.py              Phase 2 — the environment (reset/step)
├── crnt_verifiers_env.py    Phase 2 — Prime Intellect Hub adapter (optional)
├── demo.py                  narrated walkthrough — run this first
├── evaluate.py              score a real model
├── ENVIRONMENT_SPEC.md      design rationale for Phase 2
├── QUICKSTART.md            this file
├── README.md
├── requirements.txt
└── requirements-env.txt
```

Then:

```bash
git add . && git commit -m "Add Phase 2: RL environment, demo, and evaluation harness"
git push
```

## Common problems

**`ModuleNotFoundError: No module named 'crnt_solver'`**
The files import each other, so they must sit in the same folder. They no
longer require you to be *inside* that folder (fixed) — but they do all
have to be in one place.

**`ModuleNotFoundError: No module named 'sympy'`** → `pip install sympy`

**Generating a task feels slow (a few seconds)**
Normal, and bounded. Symbolic solving occasionally hits a slow case, so
every task-generation call has a 4-second wall-clock budget and falls back
to a simpler network rather than hanging. Higher `--difficulty` is slower.

**`evaluate.py` says no API key** → it's telling you exactly what to do,
or run `demo.py` instead, which needs nothing.
