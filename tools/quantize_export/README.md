# `quantize_export/` — model export and quantization

Produce the ONNX models the prover consumes: fp32 exports (no KV-cache, fixed shapes) and static
per-tensor int8 (QDQ) versions, plus a few (de)quantization utilities. Outputs land under `model/`.

## Export (HF / PyTorch → prover-ready ONNX)

| File | Purpose |
|---|---|
| `export_gpt2_nocache.py` | Export GPT-2 small (12L, `use_cache=False`) to a no-cache fp32 ONNX. |
| `export_llm_onnx.py` | Export an HF causal-LM (Qwen / Llama / Gemma / GPT-2) to a no-cache fp32 ONNX. |
| `export_bert.py` | Export BERT-base (input-ids-only, fixed seq) to fp32 ONNX + static-QDQ int8. |
| `export_cnn_small.py` | Export a small random-weight residual CNN (`model/cnn_small.onnx`) for the ml-float graph test. |
| `export_quarot_onnx.py` | Export a QuaRot-rotated GPT-2 ONNX the prover can load directly. |
| `fetch_eval_data.py` | Download the GPT-2 text eval sets into `model/eval/gpt2_eval/`. |

## Quantize / dequantize

| File | Purpose |
|---|---|
| `quantize_cnn.py` | Download (or take local) an ONNX CNN and static-QDQ-quantize it to int8. |
| `quantize_gpt2_qdq.py` | Static-QDQ int8 quantize the clean fp32 12-layer GPT-2. |
| `quantize_llm_qdq.py` | Static-QDQ int8 quantize an exported LLM (calibrated on real text). |
| `requant_cnn_imagenet.py` | Re-quantize a CNN using ImageNet calibration (spread across classes). |
| `dequant_qdq_to_fp32.py` | Rebuild/rewire QDQ nodes back to fp32 (strip quantization). |

`export_quarot_onnx.py` imports `from tools.rotate_fold.quarot_gpt2 import …`; run from the repo root
with `PYTHONPATH=.`.
