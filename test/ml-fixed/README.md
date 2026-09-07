# ml-fixed provers (GPT-2, BERT & CNNs)

VOLE-ZK proofs of a **uniform fixed-point model forward pass** through the ml-fixed graph prover. Unlike
ml-quant (per-tensor int8/int12), ml-fixed puts every weight and activation on a **single uniform scale
2⁻ᴾ** (DeepProve "Case 1"): a rescale is a rounded power-of-two shift, so there are **no per-tensor scales
and no calibration**. The prover (`ALICE`) proves to the verifier (`BOB`) that it ran the model on public
inputs, committing the dense weights as a private witness.

| model | driver | notes |
|---|---|---|
| **GPT-2** | [`gpt2_prove.cpp`](gpt2_prove.cpp) | proven argmax next-token per position |
| **BERT** | [`bert_prove.cpp`](bert_prove.cpp) | Tanh pooler + classifier argmax |
| **CNN / MLP** | [`onnx_infer.cpp`](onnx_infer.cpp) | graph-driven; built-in bit-exact check vs a plaintext reference |

`bench_full.cpp` / `bench_micro.cpp` are synthetic-weight timing benchmarks (whole net / one block), not
real-model provers. All three real provers take an **fp32 ONNX directly** — no rotation, no QDQ export; the
shape machinery is constant-folded + fused at load. (For byte-identical graphs across machines — e.g. a
distributed pair — you can pre-fold once with `fullfold_onnx.py`; see §2.) Each section
below covers all three, calling out per-model differences inline.

---

## 1. Build

Prereqs: the [vole-zk](https://github.com/vole-labs/vole-zk) submodule built into `external/install`
(see the [root README](../../README.md)), OpenSSL, CMake ≥ 3.21, a C++20 compiler with AES-NI/AVX.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/external/install"
cmake --build build -j --target test_ml-fixed_gpt2_prove    # GPT-2
cmake --build build -j --target test_ml-fixed_bert_prove    # BERT
cmake --build build -j --target test_ml-fixed_onnx_infer    # CNN / MLP
# binaries land in build/bin/
```

---

## 2. Prepare the model

ml-fixed proves an **fp32 ONNX as-is** — the weights are committed and quantized internally to the uniform
2⁻ᴾ grid. There is **no QuaRot, no QDQ, and no fixed-seq fold step**; the driver constant-folds + fuses the
graph when it loads.

- **GPT-2** — `model/gpt2-12_nocache/model.onnx` (**real 12-layer GPT-2**, HF no-cache export) is the bare-run
  default. **For cross-arch or distributed runs, pre-fold the model first** (box below).
- **BERT** — `model/bert_base_fp32_basic.onnx` (encoder + Tanh pooler + 2-class head; the classifier is
  random-init, so BERT is a **path test**, not an accuracy result).
- **CNN / MLP** — any fp32 ONNX: `model/eval/resnet18-v1-7/resnet18-v1-7.onnx`, `resnet50`, `vgg16`, or the
  small default MLP. `onnx_infer` aborts with the op name if the graph uses an unsupported operator.

> **Cross-arch note (arm64 ↔ x86).** The load-time fold's out-of-range `double → int64` conversions
> (`Slice` ends = `INT64_MAX`, `Cast` of the causal-mask `−∞`) were undefined behavior and used to diverge
> between arm64 and x86; the fold now saturates these explicitly, so **the raw fp32 models fold identically
> on both architectures**. For a distributed pair you can still **Python-pre-fold once** to guarantee both
> hosts load a byte-identical graph:
> ```bash
> python3 tools/fullfold_onnx.py model/gpt2-12_nocache/model.onnx model/gpt2-12-folded-s512.onnx --seq 512
> ```
> Then pass the folded file as `MODEL=`.

---

## 3. Input data

- **GPT-2** — public token ids via `IDS` (comma-separated); length = the `seq` CLI arg. Default `100+i`.
- **BERT** — `input_ids` via `IDS`, or `SEQ` env for the length (default 16); default ids `101+i*7`.
- **CNN / MLP** — `INPUT_F32=<raw float32 CHW file>` loads a real input (else a default one). The model
  path is the **3rd CLI argument** (see §5).

`attention_mask`/`position_ids` are injected as public constants for the transformers.

---

## 4. Precision (`P`) — no calibration needed

ml-fixed has **no calibration file**: the uniform 2⁻ᴾ scale is fixed for the whole graph. The one accuracy
knob is **`P`** (fractional bits). **`P=12` is ≈fp32**; the low defaults (GPT-2 `P=7`) are coarse and only
useful for quick smoke tests. Higher `P` → more accurate but wider range-checks. Set it with the `P` env
var — nothing else to prepare.

---

## 5. Run the proof

Args: `<binary> <party> <port> <seq | model>` — `party` 1 = prover/ALICE (listens), 2 = verifier/BOB.
BOB connects to `PEER_IP` (default 127.0.0.1). **All three drivers support a distributed split** — run
BOB on a second host with `PEER_IP=<prover IP>` (same `MODEL`/`SEQ`/`IDS` on both parties).
Threads use ports `port … port+ZK_THREADS-1`.

```bash
# --- GPT-2 ---   <party> <port> <seq>   (real GPT-2; pre-folded model for cross-arch/distributed — see §2)
E="P=12 ZK_THREADS=16 LSE=1 BENCH=1 MODEL=model/gpt2-12-folded-s512.onnx"
env $E build/bin/test_ml-fixed_gpt2_prove 1 24000 512 > a.log 2>&1 &  sleep 4
env $E build/bin/test_ml-fixed_gpt2_prove 2 24000 512 > b.log 2>&1
# distributed: run party 2 on a second host with PEER_IP=<party-1 IP> (same MODEL/IDS/seq)

# --- BERT ---   <party> <port>   (length comes from the SEQ env, NOT a positional arg)
E="P=12 SEQ=512 LSE=1 BENCH=1 MODEL=model/bert512_fp32.onnx"   # or SEQ=16 + bert_base_fp32_basic.onnx
env $E build/bin/test_ml-fixed_bert_prove 1 24010 > a.log 2>&1 &  sleep 4
env $E build/bin/test_ml-fixed_bert_prove 2 24010 > b.log 2>&1

# --- CNN / MLP ---   <party> <port> <model>
M=model/eval/resnet18-v1-7/resnet18-v1-7.onnx
P=12 build/bin/test_ml-fixed_onnx_infer 1 24020 $M > a.log 2>&1 &  sleep 4
P=12 build/bin/test_ml-fixed_onnx_infer 2 24020 $M > b.log 2>&1
```

- **`LSE=1`** — DeepProve LogSumExp softmax: drops the max-subtract (`select_max`) and the division, and
  batches the exp lookup. Recommended for GPT-2/BERT; the division softmax is the default without it.

---

## 6. Reading the output (prover / ALICE side)

```
[BENCH]  ... end-to-end  ...s  ...MB comm  peakRSS ...MB
pass: ml-fixed proved ... (verifier accepted)          # <- success (BOB prints "verifier: ... proof complete")
```
- **GPT-2** prints `predicted next-token ids (proven argmax): ...`.
- **BERT** prints `argmax class=...` (the 2-class head); `NO_ARGMAX=1` reveals the raw logits instead.
- **CNN / MLP** reveals every output and checks it **equals the plaintext integer reference** computed
  inline (`matches plaintext reference`); a mismatch aborts with `graph inference mismatch`. It also prints
  `ARGMAX (predicted class)`.

---

## 7. Experiment knobs (env vars)

| var | meaning |
|---|---|
| `P` | fixed-point fractional bits (uniform 2⁻ᴾ). `12` ≈ fp32; low values are coarse. |
| `LSE` | DeepProve LogSumExp softmax (no `max_value`, no division). Needs no extra setup. |
| `MODEL` | path to the fp32 ONNX (GPT-2/BERT; CNN takes the model as the 4th CLI arg instead). |
| `IDS` / `SEQ` | GPT-2/BERT public token ids / sequence length. |
| `INPUT_F32` | (CNN) raw float32 CHW input file to prove on (else a default input). |
| `ZK_THREADS` | worker threads (GPT-2 / `onnx_infer`; BERT is fixed at 4). |
| `PEER_IP` | (party 2 only) the prover's address to connect to — set for a distributed run (default 127.0.0.1). |
| `SCORE_BITS` | (GPT-2) override the softmax exp-table width (default `P+9`); rarely needed. |
| `NO_ARGMAX` | (BERT) reveal the raw class logits instead of the proven argmax. |
| `OREF_DUMP` | (CNN) dump the plaintext reference output to a file. |
| `BENCH` | per-phase wall-time + communication. |

---

## 8. Accuracy & validation

- ml-fixed is a uniform 2⁻ᴾ scheme; at **`P=12` it is ≈fp32** — and unlike ml-quant it needs **no rotation
  and no per-tensor calibration**, ever.
- **CNN / MLP proofs self-verify**: `onnx_infer` checks the revealed ZK output is **bit-exact** against the
  plaintext integer reference the handler computes inline, so a passing run proves the fixed-point circuit
  matches the reference exactly.
- **ImageNet accuracy sweep** — `tools/imagenet_accuracy_sweep.py all <N>` reports the ml-fixed column
  alongside fp32 and ml-quant on real val images.
- **Softmax** — the `LSE=1` and default division paths give the same argmax; LSE only changes *how* the
  softmax is proven (see the ml-quant README §8 for the equivalent mimic-based check).
