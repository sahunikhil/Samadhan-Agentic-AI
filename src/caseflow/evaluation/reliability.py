"""Reliability metrics for nondeterministic agents: pass@k vs pass^k.

Run each scenario ``n`` times; ``c`` = successful trials.

* **pass@k** (Chen et al. 2021, code generation): probability that *at least one* of k samples
  succeeds = ``1 - C(n-c, k) / C(n, k)``. Right when a human/verifier can pick the good sample.
* **pass^k** (Yao et al. 2024, tau-bench): probability that *all* k trials succeed =
  ``C(c, k) / C(n, k)``. Right for customer-facing agents: every customer gets exactly one try,
  so an agent that solves a task 3 times out of 4 fails one customer in four. pass^k falls
  quickly with k when behavior is inconsistent - the number that matters for production.

Both are unbiased estimators from n >= k trials (not just "run k times once").
"""

from __future__ import annotations

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (n={n}, c={c}, k={k})")
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (n={n}, c={c}, k={k})")
    return comb(c, k) / comb(n, k)


def reliability_summary(successes: dict[str, list[bool]]) -> dict[str, float]:
    """Mean pass^k and pass@k over scenarios, for every k up to the trials run per scenario."""
    if not successes:
        return {}
    n = min(len(v) for v in successes.values())
    out: dict[str, float] = {}
    for k in range(1, n + 1):
        out[f"pass^{k}"] = round(sum(pass_hat_k(n, sum(v[:n]), k) for v in successes.values()) / len(successes), 4)
        if k > 1:
            out[f"pass@{k}"] = round(sum(pass_at_k(n, sum(v[:n]), k) for v in successes.values()) / len(successes), 4)
    return out
