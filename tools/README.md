# `tools/` — plaintext utilities for the emp-zkml ZK prover

Python helpers around the ZK inference prover: they export and quantize models, reproduce the
prover's arithmetic bit-for-bit in plaintext, measure accuracy/agreement across precision modes, and
prepare graphs (rotation, folding). None of these run the ZK protocol itself — they are the
plaintext side used to build inputs for the prover and to validate its outputs.

## Layout

| Folder | What's in it |
|---|---|
| [`accuracy/`](accuracy/) | Top-1 accuracy and per-input agreement sweeps across fp32 / fixed-point / quant (the source of the paper's §4.3 tables) |
| [`mimics/`](mimics/) | Bit-exact plaintext mimics that reproduce each prover scheme (`ml-fixed`, `ml-quant`, `ml-float`) so a mimic's output equals the proof's output |
| [`quantize_export/`](quantize_export/) | Export HF/ONNX models to the prover's fp32 / QDQ-int8 formats, and (de)quantize them |
| [`rotate_fold/`](rotate_fold/) | QuaRot rotation and ONNX graph folding/fusing helpers used before proving |
| [`infra/`](infra/) | Network shaping for distributed WAN benchmarks |

Each subfolder has its own `README.md` with a per-file description.

## Conventions

- **Run from the repository root** with `PYTHONPATH=.` (e.g., `PYTHONPATH=. python3 tools/accuracy/gpt2_llm_windows.py 512 10`). Data/model paths inside the scripts (`model/…`, `model/eval/…`) are repo-root-relative.
- **Cross-folder use:** scripts import siblings by dotted path, e.g. `import tools.mimics.prover_exact_mimic`, `from tools.rotate_fold.quarot_gpt2 import load`. A few accuracy scripts also `exec(open('tools/mimics/plaintext_*.py'))`. These resolve as namespace packages (no `__init__.py` needed) when run from the repo root.
- **Dependency direction:** `accuracy/` → `mimics/` (+ `rotate_fold/quarot_gpt2`); `mimics/` → `rotate_fold/quarot_gpt2`. `quantize_export/` and `rotate_fold/` are mostly standalone.
- **Python packages:** `numpy`, `onnx`, `onnxruntime`, `transformers`, `pillow`, `scipy` (no PyTorch for the accuracy/mimic path).
- Non-`.py` helpers (`fetch_models.sh`, `build_deps.sh`) stay at this top level.
