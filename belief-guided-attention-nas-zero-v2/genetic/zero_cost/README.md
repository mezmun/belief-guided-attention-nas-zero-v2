# Passive zero-cost proxy bank — V2

Zero-cost measurements are observational in this version. They do not alter
candidate generation, belief scores, search selection, fitness, survivor
selection, or similarity learning.

The configured proxies are calculated before real training for all 20 search
candidates and for the independent audit candidate. After training, true
fitness is attached to the same records so proxy ranking can be evaluated.

## Included methods

- `synflow`
- `snip`
- `gradnorm`
- `plain`
- `l2norm`
- `fisher`
- `grasp`
- `jacov`
- `zico`
- `zen`
- `naswot`
- `swap`
- `meco`
- `macs`
- parameter count as a separate baseline

## Outputs

`candidate_scores.csv` and `evaluated_scores.csv` contain
`evaluation_role=search|audit`, `candidate_source`, belief values and proxy
values. Cycle metrics are reported separately for the search subset, audit
subset when meaningful, and all observed evaluations. `proxy_health.csv`
records proxy success/runtime and `proxy_pairwise_correlations.csv` records
pairwise signal correlations.

Cycle files are replaced idempotently on restart rather than appended twice.

## Test

```bash
python -m genetic.zero_cost.self_test
```

The self-test uses a synthetic CNN and does not require CINIC-10.
