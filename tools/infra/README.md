# `infra/` — benchmark network shaping

Utilities for the distributed (two-party) WAN benchmarks: shape link bandwidth and latency so
prover/verifier runs reflect realistic network conditions.

| File | Purpose |
|---|---|
| `throttle.py` | Set WAN bandwidth / latency via Linux `tc` (`tbf` + `netem`). Note: `-l` is **half** the RTT (60 ms RTT → `-l 30`). |
| `throttle_macos.py` | macOS counterpart of `throttle.py`. |
