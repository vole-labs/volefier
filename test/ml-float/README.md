# ml-float provers (Mystique: field-linear + IEEE-754 float non-linear)

VOLE-ZK proofs of a model forward pass on the **ml-float** backend. Unlike the field-only lookup tracks
([`ml-quant`](../ml-quant), [`ml-fixed`](../ml-fixed)), ml-float runs the **linear layers in the
fixed-point field** but evaluates the **non-linear parts in genuine IEEE-754 floating-point Boolean
circuits** (the Mystique approach): activations are bridged between the arithmetic field and the
bit/float domain (emp-zk `edaBit` + Mystique `Fix<->Float`) at each boundary. This is the **accurate-but-slow,
compute-bound** path — floats reproduce the original model exactly, but bit-level ops dominate the cost. See
the [library README](../../volefier/ml-float/README.md) for the design rationale.

| driver | what it proves | build status |
|---|---|---|
| [`onnx_infer.cpp`](onnx_infer.cpp) | graph-driven ONNX inference: field conv/fc, Boolean ReLU/MaxPool, float BatchNorm/Sigmoid/Gelu | **live** |
| [`float_convert_test.cpp`](float_convert_test.cpp) | round-trip of the ported `Int62ToFloat` / `FloatToInt62` conversion circuits | **live** |
| [`float_nonlinear.cpp`](float_nonlinear.cpp) | standalone float activation / norm / softmax vs a plaintext `double` reference | disabled* |
| [`float_convert.cpp`](float_convert.cpp) | prototype arith↔float round-trip at one layer boundary (edaBit + Fix↔Float, ReLU in Boolean) | disabled* |
| [`resnet.cpp`](resnet.cpp) | float-native ResNet (activations carried in float throughout) — the naive baseline, kept for comparison | disabled* |

\* Commented out in [`test/CMakeLists.txt`](../CMakeLists.txt) pending a port to the new emp API; only
`onnx_infer` and `float_convert_test` are currently built. Re-enable after migrating `volefier/ml-float/*.h`.

---

## 1. Build

Same emp submodule prereqs as the other tracks (see the [root README](../../README.md)):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/external/install"
cmake --build build -j --target test_ml-float_onnx_infer
cmake --build build -j --target test_ml-float_float_convert_test
# binaries land in build/bin/
```

## 2. Run

Args: `<binary> <party> <port> [model.onnx]` — `party` 1 = prover/ALICE (listens), 2 = verifier/BOB
(connects to `PEER_IP`, default `127.0.0.1`). The Boolean COT needs a **writable `./data/`** in the run
directory. `onnx_infer` takes the fp32 ONNX as the **3rd CLI arg** (default: a small DeepProve asset).

```bash
# --- ONNX inference ---   <party> <port> <model>
M=model/eval/resnet18-v1-7/resnet18-v1-7.onnx
ZK_THREADS=16 build/bin/test_ml-float_onnx_infer 1 12345 $M > a.log 2>&1 &  sleep 4
ZK_THREADS=16 build/bin/test_ml-float_onnx_infer 2 12345 $M > b.log 2>&1
# distributed: run party 2 on a second host with PEER_IP=<party-1 IP> (same model)

# --- conversion-circuit unit test ---   <party> <port>
build/bin/test_ml-float_float_convert_test 1 12345 &  sleep 1
build/bin/test_ml-float_float_convert_test 2 12345
```

The ONNX prover reveals every output, prints the predicted `ARGMAX`, and checks the **max-abs-error vs a
plaintext `double` reference** stays within a float ULP tolerance (it aborts if too large — floats aren't
bit-exact). Success prints `pass: ml-float proved ONNX inference (…)` on ALICE and
`verifier: ml-float proof complete` on BOB. Aborts with the op name on an unsupported operator.

## 3. Knobs (env vars)

| var | meaning |
|---|---|
| `ZK_THREADS` | worker threads (default 4). |
| `COT_THREADS` | Boolean-COT threads (default = `ZK_THREADS`). |
| `PEER_IP` | (party 2 only) prover's address for a distributed run (default `127.0.0.1`). |
| `INPUT_F32` | (`onnx_infer`) raw float32 CHW input file to prove on (else a default input). |
| `OREF_DUMP` | (`onnx_infer`) dump the plaintext reference output to a file. |
