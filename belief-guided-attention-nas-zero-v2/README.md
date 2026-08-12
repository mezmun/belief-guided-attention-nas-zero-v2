# Belief-Guided Attention NAS V2

Evolutionary neural architecture search for attention-aware CNN architectures with
belief-guided candidate ranking and passive zero-cost monitoring.

## V2 search flow

- Generations 1-5 use the warm-up evolutionary search.
- Guided generations construct 100 unique, previously unseen candidates:
  - 80 evolutionary candidates
  - 15 balanced candidates
  - 5 novelty-injected candidates
- The 20 search evaluations are selected as:
  - 15 belief-mean exploitation
  - 4 UCB selections
  - 1 bounded-novelty selection
- One additional random audit candidate is sampled from the candidates rejected by
  the search selector. The audit receives passive zero-cost measurements and full
  training, but it is excluded from the belief archive, similarity learning,
  environmental selection, generation mean, and future search decisions.

## Similarity and belief

V2 starts each fresh run with equal similarity-component weights and learns them
online only from that run's search evaluations. The components are:

1. module sequence
2. base sequence
3. attention sequence
4. count similarity
5. module bigram similarity
6. normalized relative-position similarity
7. structural numeric similarity
8. capacity numeric similarity

Belief neighbour evidence uses the nearest 20 eligible evaluated architectures (or all
available entries when the archive contains fewer than 20). Kernel weighting is
applied inside this local neighbourhood; the configured Bayesian estimator also keeps
an archive-level prior for shrinkage.

## Diversity

Guided candidate generation enforces architecture-level uniqueness and excludes
architectures already present in the active search archive/cache. Environmental
selection also deduplicates architectures and samples survivors without replacement,
so the 20-parent population contains 20 unique architectures.

## Restart behaviour

Restart is model-level rather than epoch-level.

- Candidate and selected sets are checkpointed per cycle.
- A successfully completed model is written immediately to the role-specific result
  file and cache.
- If a run is interrupted during one model, completed models are reused on restart
  and only the incomplete model starts again from epoch 1.
- Search and audit caches are independent.
- Cycle-level monitoring files are replaced/upserted on restart to avoid duplicate
  rows.

## Zero-cost

The configured zero-cost proxies are passive observers in this version. They do not
change belief scores, candidate selection, novelty, uncertainty, or survivor selection.
The audit candidate is also measured so that additional out-of-selector evidence is
available for later analysis.

## Main outputs

Runtime outputs are created under:

- `populations/`
- `log/`
- `belief_outputs/run_*/`

`populations/generation_metrics.csv` records search-generation means, selection-group
means, audit fitness, candidate source counts, uniqueness counts, survivor statistics,
and best-so-far search fitness.

Belief outputs include candidate predictions, evaluated search records, cycle metrics,
selection summaries, the evaluated-architecture archive, and similarity-weight history.
Cycle metrics emphasize ranking/selection diagnostics (Spearman, Kendall tau-b, pairwise
ordering accuracy, best-model rank, top-k recall, selected top-k mean and top-k mean
regret); Pearson/MAE/RMSE remain secondary diagnostics. These ranking metrics are
computed only on the search-evaluated subset because rejected candidates have no truth.
Zero-cost outputs are stored below the active run's `zero_cost/` directory.
