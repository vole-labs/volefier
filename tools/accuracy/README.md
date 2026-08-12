# `accuracy/` — accuracy and agreement sweeps

Measure **top-1 accuracy** (vs gold label) and **per-input top-1 agreement** between fp32,
uniform fixed-point (`ml-fixed`), and quantized inference. These produced the secure-ZKML §4.3
tables. They import the mimics in [`../mimics/`](../mimics/); run from the repo root with `PYTHONPATH=.`.

## CNN (ResNet-50, VGG-16, ImageNet)

| File | Purpose |
|---|---|
| `fixed_vs_quant_psweep.py` | Table-6 sweep: fp32 + ZK-int8 + fixed-`P`; accuracy, `fix↔fp32`, `fix↔quant`. OFFSET/STRIDE/IMG range to combine disjoint image sets. |
| `fixed_vs_stdquant_psweep.py` | Adds a **standard int8** baseline (ORT on the QDQ model); produced **Table 7** (`fix↔{fp32,std,zk}`, `zk↔std`). Arg `N` = number of classes (100×3 = 300-image sample). |
| `cnn_accuracy_sweep.py` | Generalized 3-way ImageNet top-1 / agree-fp32 sweep (fp32 vs ml-fixed vs ml-quant). |
| `imagenet_accuracy_sweep.py` | ImageNet accuracy sweep (per-class subselection). |
| `fixed_precision_sweep.py` | ml-fixed top-1 / agree-fp32 as a function of precision `P`. |
| `accuracy_baselines.py` | 3-way int8 comparison: fp32 vs ideal-float-scale int8 vs prover `c·2^e` int8. |

## GPT-2 (WikiText-2 next-token)

| File | Purpose |
|---|---|
| `gpt2_fixed_vs_quant.py` | GPT-2 analog of Table 6 over one window: fixed-`P` vs prover-exact quant int-`B`. |
| `gpt2_llm_windows.py` | **Windowed, pooled** run (`W_LEN`×`K` windows) that produced the finalized GPT-2 table. Usage: `gpt2_llm_windows.py [W_LEN=512] [K=10]`. |
| `gpt2_fix_vs_std.py` | The **`fix↔std` column**: fixed forwards + standard int8 (ORT `quantize_dynamic`), pooled; rechecks `fix↔fp32`. Env `STD=<int8.onnx>`. |
| `gpt2_accuracy_sweep.py` | GPT-2 fp32-baseline accuracy sweep (ml-fixed vs prover-exact quant). |
| `gpt2_argmax_accuracy.py` | GPT-2 `ml-quant` per-tensor int-N argmax-accuracy harness. |
| `gpt2_ptq_baselines.py` | GPT-2 post-training-quantization baselines (argmax agreement vs fp32). |

## Notes

- "Completeness error" in the paper = `100% − (fix↔quant)`.
- **std-int8 GPT-2** is not a script here — build it with ORT:
  `python3 -c "from onnxruntime.quantization import quantize_dynamic,QuantType; quantize_dynamic('model/gpt2-12_nocache/model.onnx','gpt2_int8_dyn.onnx',weight_type=QuantType.QInt8)"`.
- fp32 = ONNX Runtime on the original model; std-int8 (CNN) = ORT on the QDQ model.
