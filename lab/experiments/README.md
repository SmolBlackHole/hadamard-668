# Completed experiment drivers

Parent: [Research laboratory](../../docs/laboratory.md)

These drivers are retained for reproducibility, not used by the reusable tools.
They intentionally contain fixed batch IDs, seeds and historical baseline logic.
Run them only when reproducing the corresponding report: they write to the fixed
`runs/research/landscape/` directory and can replace its previous report files.

| Driver | Purpose | Results |
| --- | --- | --- |
| `python -m lab.experiments.replay` | Explain the 6M/10M n=56 discrepancy | [Landscape results](../../docs/landscape-results.md) |
| `python -m lab.experiments.landscape` | Matched-state and perturbation recovery controls | [Landscape results](../../docs/landscape-results.md) |
| `python -m lab.experiments.greedy` | Compare the recorded scan with the full-scan prototype | [Landscape results](../../docs/landscape-results.md) |

The original greedy baseline is extracted from `bb26d1d`; retain that Git history.
Historical phase functions are passed explicitly through `SearchOperators` for
each run, including nested quenches. They do not replace global solver functions.
New algorithm development uses separate bounded experiments, followed by
direct production integration only after explicit evaluation.
