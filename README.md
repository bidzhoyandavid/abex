# abex

Library for analyzing A/B test results. See `ABTEST_LIB_SPEC.md` for the
full design spec.

## Setup

```
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` is required after cloning — commits run `pytest tests/`
locally before they're accepted.

## Status

Implemented: `data/`, `design/`, `stats/frequentist.py`, `stats/bootstrap.py`,
`stats/multiple_testing.py`, `stats/ratio.py`, `analysis/effect_size.py`,
`analysis/guardrails.py`, `report.py`, `selector/` (wired only to the
implemented stats functions above).

`stats/ratio.py` linearizes ratio metrics (revenue/user, CTR with repeated
impressions, ...) into a per-cluster continuous value — feed the result into
`stats/frequentist.py` or `stats/bootstrap.py` as usual, no separate ratio
test implementation needed.

Stubbed (raise `NotImplementedError`, not wired into the selector registry):
`stats/bayesian.py`, `stats/sequential.py`, `stats/cuped.py`,
`analysis/segments.py`, `analysis/novelty.py`, `viz/*`.
