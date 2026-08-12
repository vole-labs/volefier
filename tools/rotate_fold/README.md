# `rotate_fold/` — rotation and graph folding helpers

Graph-level preprocessing applied before proving: QuaRot outlier-taming rotations, and ONNX folding
/ fusing / renaming so the prover's front end and fast handlers pick the right ops. Producing a
deterministic, prover-friendly graph.

## QuaRot rotation

| File | Purpose |
|---|---|
| `quarot_gpt2.py` | QuaRot residual-stream rotation for GPT-2 (one orthonormal Q). **Imported by** `mimics/prover_exact_mimic.py`, several accuracy scripts, and `export_quarot_onnx.py`. |
| `quarot_qwen.py` | QuaRot residual rotation for Qwen2.5 (plaintext validation). |

## ONNX folding / fusing

| File | Purpose |
|---|---|
| `prefold_scale.py` | Pre-fold GPT-2 attention-scale `Mul` nodes (`1/√d`) into weights. |
| `bake_bert_scale.py` | BERT twin of `prefold_scale.py` (attention-scale bake). |
| `fullfold_onnx.py` | Freeze `input_ids` shape and bake aux inputs (`attention_mask=ones`, `position_ids=arange`) for cross-architecture determinism. |
| `fuse_llm_onnx.py` | Fuse decomposed LLM subgraphs back into single ops the ml-quant/ml-fixed front end handles. |
| `name_hadamard.py` | Rename the shared online-Hadamard weight so the prover's fast-WHT handler detects it. |
