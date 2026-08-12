# V2 change log

## Search and diversity

- Guided pool remains 80 evolutionary + 15 balanced + 5 novelty-injected = 100.
- Guided evaluation allocation is 15 belief-mean + 4 UCB + 1 bounded novelty.
- One additional independent random audit is trained outside the 20 search evaluations.
- Environment selection deduplicates by architecture and samples survivors without replacement.
- Parent/candidate/survivor uniqueness and source-specific generation metrics are recorded.

## Similarity and belief

- Fresh runs use equal cold-start similarity weights.
- Removed redundant base-attention-pair bigram/count duplication.
- Added normalized relative-position similarity.
- Split numeric similarity into structural and capacity components.
- Belief prediction now actually uses the nearest 20 eligible archive architectures.
- Similarity-pair sampling is deterministic and weight history is stored per cycle.

## Restart and data integrity

- Candidate, selected, and evaluated cycle checkpoints were added.
- Resume is model-level: completed models are reused; an interrupted model restarts from epoch 1.
- Search and audit caches/results are separate.
- Candidate generation and environment selection use deterministic cycle-local random streams.
- Runtime CSV outputs use cycle-level replace/upsert semantics to avoid duplicate restart rows.
- Generation/individual IDs use the actual current cycle.
- Fresh runs clear prior runtime caches/checkpoints/generated individuals before initialization.

## Zero-cost

- All configured zero-cost proxies remain passive.
- The independent audit candidate also receives zero-cost measurements.
- Search/audit roles are retained in zero-cost output tables.

## Evaluation metrics

- Ranking/selection diagnostics are primary: Spearman, Kendall tau-b, pairwise ordering accuracy, best-true-model belief rank, top-k recall, selected/true top-k means, and top-k mean regret.
- Pearson, MAE, and RMSE remain secondary diagnostics.
- Guided-cycle ranking metrics are explicitly restricted to the 20 search-evaluated candidates; the 100-candidate pool cannot be fully ranked without additional truth.
