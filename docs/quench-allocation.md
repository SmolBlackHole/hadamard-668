# Quench allocation experiment

Parent: [Documentation index](README.md)

## Contents

- [Quench allocation experiment](#quench-allocation-experiment)
  - [Contents](#contents)
  - [Question and decision](#question-and-decision)
  - [Method](#method)
  - [Results](#results)
  - [Limits and next question](#limits-and-next-question)

## Question and decision

Does splitting targeted escape into shorter nested searches improve solution yield
at the same total candidate budget? This screening experiment found no increase
in total solutions. Keep the solver defaults unchanged.

## Method

The solver source is unchanged from commit `cca9a37`. The research branch adds
quench-budget and seed-start options to the existing ablation runner. Each arm uses
the same 1,000 seeds (42 through 1041) at lengths 46, 50 and 54, random starts and
6,000,000 candidate evaluations per run. All other settings retain their defaults.

```bash
python -m lab.compare --quench-budget 100000 --quench-budget 500000 \
  --n 46 --n 50 --n 54 --seeds 1000 --seed-start 42 \
  --candidate-budget 6000000 --workers 12 \
  --output runs/quench-budget-screen.json
```

The run used Windows and Python 3.13.14 with the repository's locked dependencies.
Solved candidates passed pipeline verification. The default arm reproduced the
solution counts of the earlier CLI sweep at these lengths.

The local raw report is `runs/quench-budget-screen.json`, SHA-256:

```text
2e76ca5ebe41a4d154e7a6140e17222bdb97254a4f130d45c353cac1f78571c5
```

The raw report is an ignored local artifact, not bundled with a clone. The command
above regenerates observations; elapsed-time fields make its hash run-specific.

## Results

Each cell shows solutions out of 1,000 paired instances.

| Configuration | n=46 | n=50 | n=54 | Total out of 3,000 |
| --- | ---: | ---: | ---: | ---: |
| Default (10,000,000 quench limit) | 3 | 1 | 0 | 4 |
| Targeted escape disabled | 4 | 0 | 0 | 4 |
| 100,000 quench limit | 4 | 0 | 0 | 4 |
| 500,000 quench limit | 2 | 0 | 0 | 2 |

The mean numbers of nested searches per run were 0.859, 0, 40.366 and 8.430,
respectively. Shorter limits successfully increased the number of attempts, but
did not increase the observed total solution count. The numbers of Q=1 endpoints
were 128, 105, 124 and 113, respectively.

## Limits and next question

There are too few solutions to conclude that the tied variants are equivalent or
that the 500,000 limit is generally worse. This is a screen, not an independent
confirmation experiment. No setting was selected for promotion, so no holdout run
was performed.

Quality checks ran concurrently during part of the experiment. Do not use its
runtime figures to claim a speed advantage. Configurations also ran in sequential
blocks, rather than randomized timing blocks.

The next research question is proposal quality: can four-sequence escape candidates
be selected using their complete residual change and diversity, instead of the first
five matching columns at one lag? This has not been implemented or measured here.
Lower immediate Q alone is not an acceptance criterion for that experiment.
