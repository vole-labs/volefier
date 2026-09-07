# ml-quant provers (GPT-2, BERT, Qwen / Llama / Gemma & ResNet)

VOLE-ZK proofs of a **quantized model forward pass** through the ml-quant graph prover. The prover
(`ALICE`) proves to the verifier (`BOB`) that it ran the model on public inputs, committing the dense
weights as a private witness. The LLMs (GPT-2, Qwen, Llama, Gemma) output the **proven argmax** next-token
id per position; BERT and ResNet output the proven classifier argmax.

| model | driver | scheme |
|---|---|---|
| **GPT-2** | [`quant_gpt2_prove.cpp`](quant_gpt2_prove.cpp) | symmetric int12 (`SYM_BITS`) + QuaRot |
| **BERT** | [`quant_bert_sym_prove.cpp`](quant_bert_sym_prove.cpp) | symmetric int12 (`SYM_BITS`) |
| **Qwen2.5** | [`quant_qwen_prove.cpp`](quant_qwen_prove.cpp) | symmetric int12 (`SYM_BITS`) + QuaRot; RMSNorm + RoPE + GQA + SwiGLU |
| **Llama-3.2** | [`quant_qwen_prove.cpp`](quant_qwen_prove.cpp) | same as Qwen (identical code path; `VOCAB=128256`) |
| **Gemma-3** | [`quant_qwen_prove.cpp`](quant_qwen_prove.cpp) | symmetric int12; + q/k-norm, sandwich norms, GeGLU (proves out of the box; massive-activation accuracy open) |
| **ResNet / CNN** | [`quant_onnx.cpp`](quant_onnx.cpp) | asymmetric uint8 QDQ |

All run through the same graph prover; **Llama and Gemma reuse the Qwen driver** (`quant_qwen_prove`) — no
separate binary. The **transformers** (GPT-2, BERT, Qwen/Llama/Gemma) use the symmetric-int (`SYM_BITS`)
path; the **CNNs** (ResNet/VGG) use the **QDQ** path (int8 baked into the ONNX, no calib). Each section
below covers all of them, calling out the per-model differences inline.

---

## 1. Build

The ZK back-end ([vole-zk](https://github.com/vole-labs/vole-zk)) is a **git submodule** built in-tree
(see the [root README](../../README.md)) — no global installs. Prereqs: OpenSSL, CMake ≥ 3.21, a C++20
compiler with AES-NI/AVX.

```bash
git submodule update --init --recursive && tools/build_deps.sh   # once per checkout / dep bump
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/external/install"
cmake --build build -j --target test_ml-quant_quant_gpt2_prove       # GPT-2 (transformer)
cmake --build build -j --target test_ml-quant_quant_bert_sym_prove   # BERT sym-int12
cmake --build build -j --target test_ml-quant_quant_qwen_prove       # Qwen2.5 sym-int12 (RMSNorm/RoPE/GQA/SwiGLU)
cmake --build build -j --target test_ml-quant_quant_onnx             # ResNet / CNN (QDQ)
# binaries land in build/bin/
```

---

## 2. Prepare the model

Transformers load a **fixed-sequence-length** ONNX (the seq is baked so the proven circuit is
deterministic across machines); CNNs load a **pre-quantized QDQ** ONNX.

### GPT-2 — QuaRot-rotated, fully-folded

The proven model is the **12-layer GPT-2 small** (124M). Full chain from Hugging Face:
```bash
# (0) HF -> no-cache fp32 ONNX (inputs input_ids/attention_mask/position_ids, opset 18)
python3 tools/export_gpt2_nocache.py gpt2 model/gpt2-12_nocache
# (a) QuaRot: fold a residual rotation Q into the weights, un-share + rotate the LM head
#     (-> `lm_head_quarot`).  Default is Q-only (fast); HADAMARD=1 also bakes the online MLP Hadamard.
PYTHONPATH=. python3 tools/export_quarot_onnx.py \
    model/gpt2-12_nocache/model.onnx  model/gpt2-12-quarot.onnx
# (b) Fully constant-fold at a FIXED seq (shape machinery -> initializers, only compute ops remain).
python3 tools/fullfold_onnx.py  model/gpt2-12-quarot.onnx  model/gpt2-folded-s512.onnx  --seq 512
```
- `tools/prefold_scale.py` bakes the attention `head_dim^-0.25` scale if the export traces it dynamically.
- Result: **`model/gpt2-folded-s512.onnx`**. Default is **Q-only**; the online MLP Hadamard (`HADAMARD=1`) is optional.

### BERT — fp32

```bash
python3 tools/export_bert.py 512                                    # -> model/bert512_fp32.onnx
python3 tools/export_bert.py 512 textattack/bert-base-uncased-SST-2 # fine-tuned head (for real accuracy)
```
- `bert512_fp32.onnx` → quantized internally to symmetric int12 via `sym_quant_act` + a static calib.
- BERT needs **no QuaRot rotation** — it runs on plain symmetric int12.
- **GOTCHA — set `ARGBITS<=47` (e.g. `ARGBITS=40`).** The head argmax range-checks
  `sel - logit_i` in `ARGBITS+1` bits, and the driver default is `ARGBITS=48`, which is
  itself vacuous: `bit_length 49 -> ceil(49/16)=4` 16-bit limbs `-> 2^64 >= p`, so
  `RangeCheckLU::check` aborts (`range_check.h`) *after* all ops dispatch — it looks like
  a proof failure but is just a bad default. The 2-class logits need ~25 bits, so any
  `ARGBITS` in `[~26, 47]` is both sound and correct; we use `40`.
- **Attention scale.** If the export traces `1/sqrt(head_dim)` dynamically
  (`Shape->...->Sqrt->Div`), bake it first with `python3 tools/bake_bert_scale.py
  <src> <dst>` (folds `head_dim^-0.25` into the attention Muls, DCEs the chain) —
  this is a *separate* fix from `ARGBITS` and cures a `mult-check` failure, not the
  range-check abort above. Regenerate the `CALIB` for the baked model.

### Qwen2.5 — QuaRot-rotated, fused RMSNorm/RoPE, fixed-seq

The proven model is **Qwen2.5-0.5B-Instruct** (RMSNorm + RoPE + GQA + SwiGLU), on the **static** symmetric-int12
path — input-independent per-tensor scales that reveal nothing per inference. Static calibration on Qwen needs
**QuaRot** first: its per-token activation outliers make a single static per-tensor scale too coarse (≈2/16
without rotation). Qwen's RMSNorm commutes with any orthonormal rotation, so an **R1** residual rotation folded
into the weights + an **R4** SwiGLU-intermediate **structured Hadamard** smooth the outliers so one static scale
suffices. Build the rotated fixed-seq model:
```bash
# (a) QuaRot rotate: R1 residual (folded into weights) + R4 SwiGLU-intermediate structured Hadamard kron(O_m,H_p2).
#     3rd arg = the FIXED seq you will prove (baked in, so the run's <seq> arg must match).
python3 tools/quarot_qwen.py export model/qwen2.5-0.5b-quarot 16
# (b) Fuse RMSNorm/SiLU/RoPE -> single ops (RMSNormalization/SiLU/RotaryEmbedding), bake position_ids/attention_mask,
#     fold -> only input_ids remains a graph input.
python3 tools/fuse_llm_onnx.py model/qwen2.5-0.5b-quarot/model.onnx model/qwen2.5-0.5b-quarot-fused.onnx
# (c) Name the shared online-Hadamard weight so the prover detects it and runs the fast butterfly-WHT path (~180x fewer ops).
python3 tools/name_hadamard.py model/qwen2.5-0.5b-quarot-fused.onnx
```
- Result: **`model/qwen2.5-0.5b-quarot-fused.onnx`** — pass this as `MODEL`, with a static `CALIB` (§4).
- Runs on the **plain symmetric int12** path (`SYM_BITS`), no QDQ. **R2** (per-head value Hadamard) is not needed
  for Qwen; the attention SCORE region is handled by `SM_CLAMP` (§5), not a rotation.
- Validated end-to-end in the ZK prover: **9/16 top-1 vs fp32** at int12, no per-inference scale leak (§8).

### Other LLMs (Llama-3.2, …) — same path

The QuaRot rotate → fuse → prove pipeline is architecture-generic for RMSNorm + RoPE + GQA + SwiGLU LLMs:
`apply_quarot` rotates weights only (leaving q/k/v bias untouched — correct with or without attention bias), so
Llama flows through the **identical** code. Only the export HF id differs:
```bash
python3 tools/quarot_qwen.py export model/llama-3.2-1b-quarot 16 unsloth/Llama-3.2-1B  # ungated mirror (meta-llama is gated)
python3 tools/fuse_llm_onnx.py model/llama-3.2-1b-quarot/model.onnx model/llama-3.2-1b-quarot-fused.onnx
python3 tools/name_hadamard.py model/llama-3.2-1b-quarot-fused.onnx
```
Run it exactly like Qwen (§4/§5) but set the model's vocab: **`VOCAB=128256`** (Llama-3.2), with `SM_CLAMP=1 SYM_BITS=12`.
- Llama-3.2-1B intermediate is 8192 = 2¹³, so the R4 Hadamard is a **pure** fast WHT (`fast_hadamard`, m=1). Plaintext
  R1+R4 static int12 = 16/16; proven end-to-end in ZK.
- **Memory:** Llama-1B is ~2.5× Qwen-0.5B — a 2-party localhost run peaks ~28 GB. It fits a 30 GB host **only** with
  `FREE_INIT=1` (frees each fp32 weight after its last use) *and* the buffering optimizations OFF (drop
  `BG_VOLE`/`IP_DEFER`/`ASYNC_SEND`), which makes it slow (~1-core, round-bound). For speed on a big model, use a
  larger instance, or commit weights (not `PUBLIC_WEIGHTS`) so `cref_` frees per-op.

### Gemma-3 — proves out of the box (graph-driven), massive-activation caveat

**Gemma-3-270m-it** (DeepProve's model) proves **end-to-end with no new handlers** — the graph-driven prover
dispatches whatever the fused ONNX contains, and every Gemma-3 feature maps onto existing gadgets:
q/k-norm & the sandwich (post-attn/post-FFN) norms & `(1+weight)` → the RMSNorm handler; GeGLU
`gelu_pytorch_tanh` → the C++ tanh-GELU fusion + the (tanh-flavor) gelu table; embed×√hidden → a constant Mul;
GQA 4:1 → Expand; per-span RoPE (local/global θ) → baked cos/sin. (Gemma-3 *removed* Gemma-2's soft-caps, so
none are needed.) This is what DeepProve's hand-built `gemma3` could not do.
```bash
FIXED_SEQ=16 python3 tools/export_llm_onnx.py unsloth/gemma-3-270m-it model/gemma-3-270m   # ungated mirror
python3 tools/fuse_llm_onnx.py model/gemma-3-270m/model.onnx model/gemma-3-270m-fused.onnx
# run like Qwen (§5) with: VOCAB=262144 SM_CLAMP=1 SYM_BITS=12
```
- **⚠ Accuracy — massive activations.** Gemma-3-270m's residual stream has a **~1e5:1** dynamic range (the
  most extreme of the LLMs here), which per-tensor int12 cannot represent — dynamic int12 ≈ 7/16 vs fp32.
  `HAD_RESID=1` (Hadamard-wrapped requant of the residual-norm inputs, §7) is sound and helps (corr 0.66→0.70),
  but full recovery needs a comprehensive rotation across the model, complicated by Gemma's sandwich norms —
  open work. Proving is solved; matching fp32 accuracy on this tiny massive-activation model is not yet.

### ResNet / CNN — pre-built QDQ (no export)

The CNN models ship pre-quantized (int8 QDQ, scales/zero-points baked in). Use the **real ImageNet-
calibrated** one:
```
model/resnet50-v1-7-qdq-imagenet.onnx   <- real calib, USE THIS
model/resnet18-v1-7-qdq.onnx  model/resnet50-v1-12-qdq.onnx  model/vgg16-12-qdq.onnx
model/resnet50-v1-7-qdq.onnx            <- synthetic calib (bad scales) -- avoid
model/eval/resnet50-v1-7/               <- fp32 reference
```
- CNNs stay on **QDQ, not the sym-int path** — the saturation clamp is load-bearing (CNNs rely on it), and
  no rotation is needed.

---

## 3. Input data

- **GPT-2** — public token ids via `IDS` (comma-separated), length must equal the folded seq
  (`position_ids`/`attention_mask` are baked):
  ```bash
  IDS=$(python3 -c "print(','.join(str(100+i) for i in range(512)))")   # benchmark ids 100..611
  ```
- **Qwen2.5** — public token ids via `IDS` (comma-separated, length must equal the baked seq), or without
  `IDS` the `<seq>` CLI arg fills defaults `100+i`. `position_ids`/`attention_mask` are baked. The
  vocabulary is `151936` (override with `VOCAB`) for the per-position argmax split.
- **Llama-3.2 / Gemma-3** — same as Qwen (`IDS`, baked positions/mask), but set the model's `VOCAB`:
  Llama-3.2 `128256`, Gemma-3 `262144`. **⚠ Eval on REAL tokenized text, not the `100+i` default** — the
  arbitrary-id defaults give un-peaked predictions that int quant flips (Llama went 3/16 → 15/16, Qwen
  9/16 → 14/16, just by switching to a real prompt).
- **BERT** — `input_ids` only, `SEQ` env (default 512). Without `IDS`, defaults to `101+i*7` (a path
  test). For real-text eval, tokenize and pass `IDS` the same way.
- **ResNet** — a preprocessed image. The driver fills a **random uint8 image** (data-independent — fine
  for timing); `INPUT_F32=<raw float32 CHW>` loads a real one (quantized to the input scale). There is no
  seq arg — the 4th CLI argument is the **model path** (see §5).

---

## 4. Calibration (static per-tensor scales — `SYM_BITS` path only)

The `SYM_BITS` transformers (GPT-2, BERT, Qwen, Llama) freeze per-tensor int12 scales from one
**calibration-write** run and reuse them input-independently. `CALIB_WRITE=<file>` records
`max|activation|` per tensor; `CALIB=<file>` then uses those scales. (Gemma uses the dynamic path — below.)

```bash
# GPT-2 example (BERT is identical with MODEL=model/bert512_fp32.onnx and no IDS):
E="ZK_THREADS=16 P=12 SYM_BITS=12 FUSED_NL=1 MODEL=model/gpt2-folded-s512.onnx IDS=$IDS"
env $E CALIB_WRITE=c512.calib  build/bin/test_ml-quant_quant_gpt2_prove 1 26400 512 &
env $E CALIB_WRITE=c512.calib PEER_IP=127.0.0.1  build/bin/test_ml-quant_quant_gpt2_prove 2 26400 512
```

For **Qwen**, write the calib on the rotated model (§2) with the Qwen flag set (`SM_CLAMP=1`, **no `FUSED_NL`**),
`<seq>` equal to the export seq (no `IDS` uses defaults):
```bash
E="ZK_THREADS=16 P=12 SYM_BITS=12 SM_CLAMP=1 MODEL=model/qwen2.5-0.5b-quarot-fused.onnx"
env $E CALIB_WRITE=qwen.calib  build/bin/test_ml-quant_quant_qwen_prove 1 26600 16 &
env $E CALIB_WRITE=qwen.calib PEER_IP=127.0.0.1  build/bin/test_ml-quant_quant_qwen_prove 2 26600 16
```

**Llama-3.2** calib is identical to Qwen (rotated model, `SM_CLAMP=1`, no `FUSED_NL`) with `VOCAB=128256`.
**Gemma-3** currently runs the **dynamic** path (no `CALIB`; per-token scales) — static calib on its extreme
massive activations is not yet solved (§2 Gemma caveat); use `HAD_RESID=1` and omit `CALIB`.

> **Regenerate the calib whenever the model, `P`/`SYM_BITS`, or the quantization policy changes.** The
> **data-movement skip** drops pass-through tensors (Reshape/Transpose/…) from the calib (GPT-2: ≈379 →
> 161 scales), so a stale calib gives a self-consistent-but-drifted result.

**ResNet/CNNs need no calib** — their int8 scales/zero-points are baked into the QDQ ONNX.

---

## 5. Run the proof

`party` 1 = prover/ALICE (listens), 2 = verifier/BOB (connects via `PEER_IP`). The engine uses **one
socket** on `port` (plus `port+1` under `BG_VOLE`). For **localhost**, party 2 uses `PEER_IP=127.0.0.1`; for **distributed**, run
ALICE on machine A and BOB on B with `PEER_IP=<A's IP>`. Localhost runs both parties on one box
(2×`ZK_THREADS` contending) → use it for correctness, not timing. The 4th CLI arg differs by path:
**transformers** take `<seq>`, **CNNs** take `<model>`.

> **Distributed / WAN benchmarking — REQUIRED host tuning.** Shape the link on both hosts with
> `sudo python3 tools/throttle.py -i <iface> -b <Mbps> -l <RTT/2>` (e.g. `-b 800 -l 30` = 800 Mb/s, 60 ms RTT;
> remove with `-d`). **You must also raise the TCP send buffer to the bandwidth-delay product.** The prover
> pushes its bulk corrections over one main socket, and the default 4 MB `net.ipv4.tcp_wmem` caps a *single*
> stream at ~half the link at 60 ms RTT (BDP ≈ 6 MB at 800 Mb/s → single stream stalls near 379 Mb/s). On
> **both** hosts, match the send buffer to the (already-tuned) receive buffer and persist it:
> ```bash
> sudo sysctl -w net.ipv4.tcp_wmem="4096 16384 33554432"                  # 33 MB max
> echo 'net.ipv4.tcp_wmem = 4096 16384 33554432' | sudo tee /etc/sysctl.d/99-zk-tcp.conf   # survive restarts
> ```
> Verify with `iperf3` that a single stream reaches the nominal rate (~795 Mb/s, not ~379). **Skip this and
> every distributed time roughly doubles** — a stop/start reverts the live sysctl, so re-apply after each restart
> unless the `/etc/sysctl.d` file is present.

```bash
# Performance envs (identical on BOTH parties; see §7 for each knob):
PERF="ASYNC_SEND=1 ASYNC_CAP_MB=2048 W_PRECOMMIT=1 MALLOC_MMAP_THRESHOLD_=536870912 MALLOC_TRIM_THRESHOLD_=536870912 \
      BG_VOLE=1 VOLE_THREADS=2 IP_DEFER=1 IP_BUF=33554432"

# --- GPT-2 (localhost) ---   <binary> <party> <port> <seq>
E="$PERF TIMING=1 ZK_THREADS=16 P=12 FUSED_NL=1 SYM_BITS=12 BENCH=1 NETIO_STATS=1 \
   MODEL=model/gpt2-folded-s512.onnx IDS=$IDS CALIB=c512.calib"
env $E build/bin/test_ml-quant_quant_gpt2_prove 1 26400 512 > a.log 2>&1 &  sleep 3
env $E PEER_IP=127.0.0.1 build/bin/test_ml-quant_quant_gpt2_prove 2 26400 512 > b.log 2>&1

# --- BERT sym-int12 --- needs CALIB (write once as in §4), plus ARGBITS=40
E="$PERF TIMING=1 NETIO_STATS=1 ZK_THREADS=16 P=12 SYM_BITS=12 FUSED_NL=1 SEQ=512 \
   MODEL=model/bert512_fp32.onnx BENCH=1 CALIB=bert_sym.calib ARGBITS=40"
env $E build/bin/test_ml-quant_quant_bert_sym_prove 1 27000 512 > a.log 2>&1 &  sleep 3
env $E PEER_IP=127.0.0.1 build/bin/test_ml-quant_quant_bert_sym_prove 2 27000 512 > b.log 2>&1

# --- Qwen2.5 sym-int12 --- rotated model + static CALIB (build in §2, write calib in §4); 4th arg <seq> = export seq.
# Qwen uses SM_CLAMP=1 and must NOT set FUSED_NL (see the two notes below the block).
E="$PERF TIMING=1 NETIO_STATS=1 ZK_THREADS=16 P=12 SYM_BITS=12 SM_CLAMP=1 \
   MODEL=model/qwen2.5-0.5b-quarot-fused.onnx BENCH=1 CALIB=qwen.calib"   # add IDS=... for real tokens (else 100+i)
env $E build/bin/test_ml-quant_quant_qwen_prove 1 26600 16 > a.log 2>&1 &  sleep 3
env $E PEER_IP=127.0.0.1 build/bin/test_ml-quant_quant_qwen_prove 2 26600 16 > b.log 2>&1

# --- Llama-3.2 --- SAME driver/flags as Qwen; just the rotated Llama model + its vocab
E="$PERF TIMING=1 NETIO_STATS=1 ZK_THREADS=16 P=12 SYM_BITS=12 SM_CLAMP=1 VOCAB=128256 FREE_INIT=1 \
   MODEL=model/llama-3.2-1b-quarot-fused.onnx BENCH=1 CALIB=llama.calib IDS=$IDS"   # ~28GB, memory-tight (§2)
env $E build/bin/test_ml-quant_quant_qwen_prove 1 26600 16 > a.log 2>&1 &  sleep 3
env $E PEER_IP=127.0.0.1 build/bin/test_ml-quant_quant_qwen_prove 2 26600 16 > b.log 2>&1

# --- Gemma-3 --- base fused model, DYNAMIC (no CALIB), + HAD_RESID; VOCAB=262144. (accuracy open, §2)
E="TIMING=1 NETIO_STATS=1 ZK_THREADS=4 P=12 SYM_BITS=12 SM_CLAMP=1 VOCAB=262144 FREE_INIT=1 HAD_RESID=1 \
   MODEL=model/gemma-3-270m-fused.onnx BENCH=1 IDS=$IDS"
env $E build/bin/test_ml-quant_quant_qwen_prove 1 26600 16 > a.log 2>&1 &  sleep 3
env $E PEER_IP=127.0.0.1 build/bin/test_ml-quant_quant_qwen_prove 2 26600 16 > b.log 2>&1

# --- CONSTANT-MEMORY LLM config (recommended) --- drop W_PRECOMMIT, add the streaming flags.
# Peak RSS becomes flat (~7-12 GB regardless of vocab/depth) instead of ~28 GB, same proof/tokens.
# For each LLM above, replace "$PERF" with the line below (keep its per-model flags: CALIB, VOCAB, HAD_RESID):
CM="ASYNC_SEND=1 ASYNC_CAP_MB=512 MALLOC_MMAP_THRESHOLD_=536870912 MALLOC_TRIM_THRESHOLD_=536870912 \
    BG_VOLE=1 VOLE_THREADS=2 IP_DEFER=1 IP_BUF=8388608 FV_DEFER=1 FREIVALDS=1 FREE_INIT=1 STREAM_LMHEAD=1 LAZY_INIT=1"
# e.g. Qwen: env $CM TIMING=1 NETIO_STATS=1 BENCH=1 ZK_THREADS=16 P=12 SYM_BITS=12 SM_CLAMP=1 VOCAB=151936 \
#            CALIB=qwen.calib MODEL=model/qwen2.5-0.5b-quarot-fused.onnx  build/bin/test_ml-quant_quant_qwen_prove <party> <port> 512
# (measured @800Mb/s,60ms: Qwen 7.4 GB, Llama 8.3 GB, Gemma 12.0 GB peak RSS -- vs 27/29/30 GB with W_PRECOMMIT and no streaming.)

# --- ResNet (QDQ) ---   <binary> <party> <port> <model>   -- always use USE_CLAMP_LUT=1 (below)
E="$PERF TIMING=1 BENCH=1 NETIO_STATS=1 ZK_THREADS=8 USE_CLAMP_LUT=1"
env $E build/bin/test_ml-quant_quant_onnx 1 28200 model/resnet50-v1-7-qdq-imagenet.onnx > a.log 2>&1 &  sleep 4
env $E PEER_IP=127.0.0.1 build/bin/test_ml-quant_quant_onnx 2 28200 model/resnet50-v1-7-qdq-imagenet.onnx > b.log 2>&1
```
- **`ARGBITS=40`** (BERT): the classifier logits are raw un-requanted accumulators (~2³³), so the argmax
  range-check needs a wider operand width than the default 24.
- **`USE_CLAMP_LUT=1`** (ResNet/QDQ): does the `[0,255]` saturation clamp as **one LogUp lookup** instead
  of two per-element comparisons. Valid when pre-clamp values fit ±2¹⁵ (well-calibrated models, e.g. the
  imagenet-calib ResNet); the synthetic-calib default can overflow it, so keep the LUT off there. **Set it
  for real ResNet.** For wider-range nets the fallback — batching the comparison clamp — is not yet
  implemented.
- **`SM_CLAMP=1` (Qwen — required).** The causal mask fills masked scores with `~-3.4e38`, which overflows
  the softmax `exp` LogUp domain. `SM_CLAMP` clamps the pre-exp key to `N-1` (masked → `exp≈0`, consistent
  with the reference). Every Qwen run — `CALIB_WRITE` and `CALIB` — needs it.
- **Do NOT set `FUSED_NL` on Qwen.** It rebuilds the softmax as GPT-2/BERT's scale-baked LSE `exp` table,
  which is incompatible with Qwen's `SM_CLAMP` path and **aborts mid-proof with a LogUp `key out of range`**
  (`zk-ram/logup_keyvalue.h`). `FUSED_NL` is a GPT-2/BERT-only knob; Qwen uses the plain (`SM_CLAMP`) softmax.
- **`CAUSAL_SM=1` (Qwen / Llama / Gemma).** The fast division-softmax path (§7): masked positions become a
  public constant (no lookup/`is_le`), valid positions drop the redundant clamp; byte-identical to
  `SM_CLAMP` alone. Add it to the LLM run blocks above (e.g. Qwen: `... SYM_BITS=12 SM_CLAMP=1 CAUSAL_SM=1 ...`).
- **⚠ `SYM_BITS=12` is load-bearing, not optional — the softmax depends on it.** Omit it and the LLM
  silently routes through the `uniform_requant` path, where the score scale is different and the masked
  softmax keys **do not overflow** the `exp` domain; there `SM_CLAMP` is a no-op and `CAUSAL_SM`'s
  self-check misfires (the mask is not causal at that scale). Always run the transformers with the full
  flag set here; if a softmax flag has no effect or a self-check misfires, verify `SYM_BITS=12` is set.

---

## 6. Reading the output (prover / ALICE side)

```
[TIMING] Add=.. Gemm=.. MatMul=.. Softmax=.. Conv=.. QuantizeLinear=.. ...   # per-op-TYPE dispatch time (NOT additive to forward)
[BENCH]  === PROVING TOTAL   ...s ...MB ===
[NETIO]  party=1 threads=.. rounds=.. sent_MB=.. recv_MB=..
pass: ml-quant proved real ... (verifier accepted)              # <- success (BOB prints "verifier: ... proof complete")
```
- `[TIMING]` is a *lens* (which op dispatch costs what), **not** an additive breakdown of forward — the
  per-op activation requant runs between dispatches.
- **GPT-2** also prints `predicted next-token ids (proven argmax): ...`.
- **Qwen2.5 / Llama-3.2 / Gemma-3** likewise print `predicted next-token ids: ...` (same driver, the proven
  argmax per position; `OREF_DUMP=<f>` dumps the raw quantized logits for an fp32 accuracy check).
- **BERT** folds its requant into `sym_quant_act` (post-op), so there is **no** `QuantizeLinear`/clamp
  line — the requant cost sits inside the per-op times, same as GPT-2.
- **ResNet/QDQ** *does* show a `QuantizeLinear=..s` line (the explicit requant + saturation clamp), which
  `USE_CLAMP_LUT=1` reduces. It also prints `ARGMAX (predicted class)` over 1000 classes and `max-abs-err`
  (0 = bit-exact vs the plaintext reference).

---

## 7. Experiment knobs (env vars)

| var | meaning |
|---|---|
| `ZK_THREADS` | worker threads (default 4). Distributed: give each party the machine's cores. |
| `P` | fixed-point fractional bits of the `2^-P` bus (benchmarks: 12; transformers). |
| `SYM_BITS` | symmetric int-N activation/weight width, zp=0 (benchmarks: 12). **Routes through `sym_quant_act`** (GPT-2, BERT); absent for QDQ CNNs. |
| `MODEL` | path to the ONNX (transformers; CNNs take the model as the 4th CLI arg instead). |
| `IDS` / `SEQ` | GPT-2/Qwen: comma-separated ids (4th CLI arg = seq, default ids `100+i`). BERT: `SEQ` length (ids default `101+i*7`, or pass `IDS`). |
| `VOCAB` | vocabulary size for the per-position argmax split. Qwen `151936` (default), Llama-3.2 `128256`, Gemma-3 `262144`. |
| `INPUT_F32` | (ResNet) raw float32 CHW image to prove on (else a random uint8 image). |
| `CALIB` / `CALIB_WRITE` | use / record the static per-tensor calibration (`SYM_BITS` path; QDQ needs none). |
| `USE_CLAMP_LUT` | (ResNet/QDQ) do the `[0,255]` clamp as one LogUp lookup vs two comparisons — big win on well-calibrated models (§5). |
| `FUSED_NL` | DeepProve LogSumExp softmax (no division) + scale-baked fused nonlinears. **GPT-2/BERT only — do NOT set on Qwen**: incompatible with its `SM_CLAMP` softmax, aborts mid-proof with a LogUp `key out of range`. |
| `SM_CLAMP` | **(Qwen / Llama / Gemma — required)** clamp the pre-`exp` softmax key to `N-1`, so the causal mask's `~-3.4e38` fill maps to `exp≈0` instead of overflowing the `exp` LogUp table. Needed by every causal LLM on the non-`FUSED_NL` softmax path; set on every run. |
| `CAUSAL_SM` | **(causal LLMs — set alongside `SM_CLAMP`)** the fast division-softmax path. WHICH keys are masked is public (causal lower-triangle), and on the `SYM_BITS` path the masked keys overflow the `exp` domain, so masked positions take the constant `exp_table[N-1]` with no lookup and no `is_le`, and valid positions look up directly (the `exp` lookup already range-checks the key `t∈[0,N)` on the verifier, so the per-element `is_le` clamp is redundant). `m` becomes a bare committed upper bound (no `select_max`); the valid-key range check pins it. Byte-identical to `SM_CLAMP` alone. Set it for every Qwen/Llama/Gemma run. Only for **square causal self-attention** (`seq_q==seq_k`); an ALICE-side self-check aborts if the model is not actually causal. Does **not** apply to the `FUSED_NL` (GPT-2/BERT LSE) path. |
| `FREIVALDS` | Freivalds-checked dense matmuls -- ON by default; set `FREIVALDS=0` to disable. |
| `FV_DEFER` | defer the per-matmul Freivalds challenge: one seed serves a whole queue of matmuls (per-job independent u/v). ON by default; `FV_DEFER=0` disables, `FV_BUF_MB` caps the queued-operand memory (default 256 MiB). |
| `ARGBITS` | argmax operand bit-width (BERT: set 40 for the raw-accumulator classifier logits). |
| `PUBLIC_WEIGHTS` | leave weights public (skip committing) — for isolating weight-commit cost. |
| `EXP_VOLE=<n>` | prepay `n` sVOLE correlations in setup (moves VOLE generation out of the forward; ignored under `BG_VOLE`). |
| `BG_VOLE` | run the sVOLE in a background producer thread on a **second socket** (`port+1`); pair with `VOLE_THREADS`. |
| `VOLE_THREADS` | sVOLE producer pool size (independent of `ZK_THREADS`; unset = same as `ZK_THREADS` -- set it explicitly with `BG_VOLE`). |
| `BG_SLOTS` | (emp-zk) sVOLE-side banked rounds in the background pipe (default 4). |
| `IP_DEFER` / `IP_BUF` | defer the IP-matmul batch checks across ops; `IP_BUF` = claim-buffer capacity (recommended 33554432). **Both parties must set the same values.** |
| `ASYNC_SEND` | (emp-tool) per-socket sender thread: `send_data` enqueues and returns, compute overlaps transmission. **Set on both parties**; pair with the `MALLOC_*` thresholds below. |
| `ASYNC_CAP_MB` | (emp-tool) async-send reservoir size (default 256): how much banked correction data can bridge compute-only stretches. RSS grows by the cap. |
| `W_PRECOMMIT` | commit all dense weights up front at run() entry, so their corrections stream through the opening compute stretch. Both parties. |
| `MALLOC_MMAP_THRESHOLD_` / `MALLOC_TRIM_THRESHOLD_` | glibc knobs (`536870912`): keep big transient buffers on the heap for reuse instead of re-mmap/zeroing them each use. |
| `FREE_INIT` | free each fp32 weight from the graph right after its last use (single-inference: mutates the loaded graph, so you can't re-prove without reload — fine per pass). Big memory win for large models (e.g. Llama-1B); pair with streaming (on by default). |
| `STREAM_LMHEAD` | **constant-memory LM head + argmax**: computes the vocabulary projection in vocab-column blocks, never materializing the `seq×vocab` logits or the whole `vocab×hidden` committed weight. Argmax uses a *public winner* (the predicted token — revealed on the non-streaming path too) with a range-checked index; the committed logits, denominator, and weight stay hidden. Untied heads only (Qwen/Llama; tied Gemma stays on `matmul_wte`). Recommended for the LLMs. |
| `LMHEAD_VOCAB_BLK` / `STREAM_ARGBITS` | (with `STREAM_LMHEAD`) vocab-block width (default `8192`) and the argmax dominance range-check width (default `44`, covers the raw accumulator). |
| `LAZY_INIT` | decode each external-data weight (`.onnx.data`) from disk only when its op runs; `FREE_INIT` frees it right after → resident fp32 is O(current op), not the whole model. Big win for external-data LLMs (Llama's 6.3 GB, committed once via a central just-in-time resolve in the dispatch loop). No effect on inline single-`.onnx` models (GPT-2/BERT). |
| `MM_ROW_BLK` / `WTE_ROW_BLK` | row-block size for the chunked `matmul` / `matmul_wte` output (default `64`): caps the transient transposed-output tensor at a large-output matmul (the LM head). |
| `ARGMAX_CHUNK` | row-chunk size (committed elements, default `~4M`) for the non-streaming `select_max_rows` argmax; bounds the `O(rows×V)` selector working set on the fallback (no-`STREAM_LMHEAD`) path. |
| `NO_TIE_WTE` | disable tied-embedding dedup. By default a fused graph whose Gather table equals the LM-head weightᵀ is committed **once** (input embedding + LM head share the cells, enforcing tied weights); set this to force a separate commitment. |
| `HAD_RESID` | Hadamard-wrapped requant for residual-stream RMSNorm inputs (`r_q = trunc(WHT(quant(WHT(pad r))))`), for massive-activation models (Gemma). Sound, committed×public; spreads per-channel outliers across the padded-Sylvester basis so int quant can represent them. Opt-in; no effect on models without the outliers. |
| `BENCH` / `TIMING` / `NETIO_STATS` | per-phase wall+comm / per-op timing / round-byte counters. |
| `PEER_IP` | (party 2 only) the prover's address to connect to. |
| `DEBUG_REVEAL`, `MIMIC_DUMP` | per-op debug reveals / node-by-node range dumps. |

---

## 8. Accuracy & validation

### GPT-2 — bit-exact plaintext mimic

[`tools/graph_mimic_gpt2.py`](../../tools/graph_mimic_gpt2.py) reproduces the GPT-2 prover
**token-for-token in numpy** (no ZK), porting the plaintext `ref_` gadgets verbatim (static calib scales,
sqrt-free int LayerNorm, LSE softmax, `scale_search` requant, data-movement skip, generic gelu). Use it to
check a scheme change without running the full proof:
```bash
grep -a "predicted next-token" a.log | head -1 | cut -d: -f2 > zk_tokens.txt
CALIB=c512.calib PYTHONPATH=. python3 tools/graph_mimic_gpt2.py zk_tokens.txt   # reports BIT-EXACT on a match
```

### BERT — SST-2

The path-test export has a random classifier head; use a **fine-tuned checkpoint** (§2) for real accuracy.
The ZK width is symmetric int12 (no rotation).

### Qwen2.5 — logit check

`OREF_DUMP=<f>` dumps the quantized logits (at `2^-P`) on the ALICE side for comparison against an fp32
reference. On a **real-text** 16-token prompt (not the `100+i` default — see §3) the proven next-token argmax
tracks fp32 at **14/16, logit corr 0.89** (rotated + static `CALIB`, int12), with input-independent scales
that leak nothing per inference (§2).

### Llama-3.2 — logit check

Same `OREF_DUMP` flow as Qwen. Real-text 16-token prompt: **15/16, corr 0.987** (QuaRot R1+R4 pure-WHT +
static calib). Proven end-to-end on the 30 GB EC2 host (2-party doesn't fit 19 GB — §2 memory note).

### Gemma-3 — proves, accuracy open

`OREF_DUMP` real-text: **7/16, corr 0.66→0.70 with `HAD_RESID`** (dynamic int12). Gemma-3-270m's ~1e5:1
massive activations aren't yet tamed to fp32-matching accuracy (§2 caveat); the proof itself is
verifier-accepted and sound. Full recovery is open work (comprehensive rotation vs. the sandwich norms).

### ResNet — ImageNet val

Run the accuracy sweep over N val images (`model/eval/imagenet/val`), comparing fp32 / ml-fixed / ml-quant:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 tools/imagenet_accuracy_sweep.py all 200
```
The ZK proof is checked bit-exact against the `plaintext_quant.py` mimic (`max-abs-err 0` on the ALICE side).
