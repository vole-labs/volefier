# `mimics/` — bit-exact plaintext mimics

Numpy reimplementations that reproduce each prover scheme **bit-for-bit**, so a mimic's output equals
what the ZK prover would output. They are used as the reference in the accuracy sweeps and to debug
the prover. Rounding must match the prover exactly: `ml-fixed` = round-half-away, `ml-quant` `c·2^e`
requant = round-half-up.

Some GPT-2 mimics import `from tools.rotate_fold.quarot_gpt2 import …`; run from the repo root with `PYTHONPATH=.`.

## CNN

| File | Mirrors | Notes |
|---|---|---|
| `plaintext_fixed.py` | `ml-fixed` (`FixedGraphInfer`) | Uniform `2^-P`; precision `P` via env `P`. |
| `plaintext_quant.py` | `ml-quant` (`QuantGraphInfer`) | Per-tensor int8; `FLOAT_SCALE` flag: `False` = prover `c·2^e` requant, `True` = idealized exact-scale int. |
| `plaintext_float.py` | `ml-float` (`FloatGraphInfer`) | IEEE-754 reference forward. |

## GPT-2

| File | Mirrors | Notes |
|---|---|---|
| `plaintext_fixed_gpt2.py` | `ml-fixed` GPT-2 | Uniform `2^-P` forward. |
| `plaintext_int8_gpt2.py` | `ml-quant` GPT-2 | Per-tensor int8 (zkGPT/DeepProve-style); idealized exact-scale. |
| `prover_exact_mimic.py` | prover's QuaRot static int-`B` | The **deployed** path: exact gadgets (`c·2^e` requant, LUT gelu, integer LayerNorm/softmax, online Hadamard). `set_bits(8/10/12)` selects the bit-width. |
| `graph_mimic_gpt2.py` | `ml-quant` **graph** prover | Graph-driven bit-exact mimic. |
| `gadget_mimic_gpt2.py` | `ml-fixed` gadgets | Gadget-faithful integer/LUT reproduction. |
| `exact_mimic_folded.py` | `ml-quant` graph prover on folded GPT-2 | Scheme-exact on `gpt2-folded-s512`. |

`prover_exact_mimic.py` is the one used for the deployed GPT-2 quant numbers in §4.3; the others are
alternative/idealized mimics kept for cross-checking.
