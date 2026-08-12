# Belief-Guided Attention NAS — V2

This package provides the similarity-based belief layer used by `evolve.py`.
The search target is candidate ranking and evaluation allocation, not direct
fitness regression.

## Guided cycle

After five warm-up cycles the guided cycle uses a 100-candidate pool:

- 80 evolutionary candidates
- 15 balanced candidates
- 5 globally novelty-injected candidates

All 100 candidates must be architecture-unique and previously unseen by the
search archive/cache.

The 20 search evaluations are allocated as:

- 15 highest belief means
- 4 highest UCB scores among candidates not already selected
- 1 bounded-novelty candidate

Bounded novelty requires novelty at or above the cycle Q75 and belief at or
above the cycle Q50, then chooses the most novel eligible candidate. If all
such candidates have already been selected, the fallback is the highest-belief
remaining candidate in the Q75 novelty group.

One additional candidate is sampled uniformly from the 80 candidates rejected
by the 20-search selection. It has `evaluation_role=audit`. The audit candidate
is trained and receives the same passive zero-cost measurements, but its truth
is not added to the belief archive, similarity learning, uncertainty
calibration, environment selection, generation mean, or best-so-far search
statistics.

## Similarity

V2 starts every fresh run from equal component weights and learns weights only
from fitness observations produced by that run. Components are:

- module sequence
- base sequence
- attention sequence
- module/base/attention count similarity
- module bigram similarity
- normalized relative-position similarity
- structural numeric similarity
- capacity numeric similarity

The redundant base-attention-pair bigram/count signal is not included as an
independent component.

## Local belief

For a new candidate, similarity is calculated against eligible archive entries,
but only the nearest `top_neighbours` entries are used in the belief update.
The V2 configuration uses `top_neighbours=20`; if fewer than 20 entries are
available, all available entries are used. Kernel weighting remains active
inside this local neighbourhood.

The default kernel is:

```text
K_ij = exp(-(1 - similarity_ij)^2 / (2 * bandwidth^2))
```

The configured `bayesian_precision` estimator uses these nearest-neighbour values
as its observation evidence and keeps an archive-level Gaussian prior for shrinkage.
Distant architectures therefore do not enter as individual kernel neighbours. The
package also records effective neighbour count, disagreement, and uncertainty for
ranking diagnostics and UCB selection.

## Persistent outputs

Each fresh run receives a separate directory:

```text
belief_outputs/
  active_run.txt
  run_YYYYMMDD_HHMMSS/
    belief_archive.json
    belief_state.json
    candidate_pre_evaluation.csv
    evaluated_offspring.csv
    cycle_metrics.csv
    selection_summary.csv
    similarity_weights_history.csv
    zero_cost/
```

Resume restores the same run directory. Search and audit results remain
separate in the runtime cache/output files. `cycle_metrics.csv` treats ranking as the
primary diagnostic and records Spearman, Kendall tau-b, pairwise ordering accuracy,
best-true-model belief rank, top-k recall, selected/true top-k means and top-k mean
regret. Pearson, MAE and RMSE are retained only as secondary diagnostics. Since only
the 20 search-selected candidates receive truth, these cycle ranking metrics describe
the evaluated subset rather than the full 100-candidate pool.

## Self-test

From the repository root:

```bash
python -m genetic.belief.self_test
```
