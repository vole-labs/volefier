# Volefier

**VOLE-ZK proofs of machine learning operations**. 
Covering both **transformers** (e.g. GPT-2, BERT, qwen, llama) and **CNNs** (e.g. ResNet, VGG).

Built over **VOLE-ZK** (QuickSilver) using [`vole-zk`](https://github.com/vole-labs/vole-zk) as the back-end, with
**LogUp lookup arguments** for the range checks and table lookups.

---

## Three representation tracks

The machine learning tasks can be implemented under three numeric representations with trade-offs between efficiency and accuracy.

| track | representation | speed and accuracy |
|---|---|---|
| [**ml-quant**](volefier/ml-quant) | **per-tensor quantized** (each tensor its own scale `c·2^e`) |  fastest but least accurate |
| [**ml-fixed**](volefier/ml-fixed) | **uniform fixed-point** (`2^-P` scale) | trade-off performance and accuracy |
| [**ml-float**](volefier/ml-float) | **IEEE-754 floating point** — Mystique | slower but full precision |

---

## Building

The ZK back-end is **[vole-zk](https://github.com/vole-labs/vole-zk)** (the emp-zk line with its
VOLE layer replaced by [vole-labs/vole](https://github.com/vole-labs/vole)), vendored as the **git
submodule** `external/vole-zk`. It carries `vole`, `emp-tool` and `emp-ot` as its own pinned
submodules and builds all of them in-tree into a local prefix (`external/install`) — no sudo, no
global installs, and every checkout pins the exact dependency commits it was validated against.

```bash
sudo apt install libssl-dev cmake build-essential          # (ubuntu; macOS: brew install openssl cmake)

git clone --recursive <this repo> && cd volefier           # or: git submodule update --init --recursive
tools/build_deps.sh                                        # builds vole-zk (+ vole, emp-tool, emp-ot) into external/install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/external/install"
cmake --build build -j
```

Requires CMake ≥ 3.21, a C++20 compiler and OpenSSL. After pulling a commit that bumps the
submodule pointer, rerun `git submodule update --init --recursive && tools/build_deps.sh` before
rebuilding.

Executables land in `build/bin/test_<folder>_<name>`.

---

## Running

Check README in `test/ml-quant/`, `test/ml-fixed/`, `test/ml-float/`

---

## License & Disclaimer

Licensed under the Apache License, Version 2.0.
Research software. Not audited; do not use to certify production inference without further review.
