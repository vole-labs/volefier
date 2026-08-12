#!/usr/bin/env python3
"""Fully constant-fold a GPT-2 ONNX at a fixed sequence length.

The C++ constant-folder has arch/build-dependent UB in its shape-arithmetic
(Shape/Gather/Reshape/Concat/Unsqueeze), so the proven circuit differs across
machines.  Instead of debugging that, we freeze the input shape to [1, seq] and
let onnxruntime's deterministic constant-folder evaluate every shape-derived
subgraph into initializers.  The result has only compute ops left, so the C++
fold is a no-op and the circuit is identical everywhere.

Output is seq-specific (one file per sequence length).
"""
import argparse, onnx, numpy as np, onnxruntime as ort
from onnx import numpy_helper
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("outp")
ap.add_argument("--seq", type=int, required=True)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--head-dim", type=int, default=64)
args = ap.parse_args()

m = onnx.load(args.inp)
# freeze input_ids shape; bake the aux inputs (attention_mask=ones, position_ids=arange)
# as constants so ORT folds the ENTIRE causal-mask path (otherwise the And/Reshape mask
# nodes survive to runtime and the C++ dispatcher can't resolve them at large seq).
keep = []
for inp in m.graph.input:
    dims = inp.type.tensor_type.shape.dim
    if len(dims) >= 1:
        dims[0].dim_value = args.batch
    if len(dims) >= 2:
        dims[-1].dim_value = args.seq
    nm = inp.name.lower()
    if "mask" in nm:
        m.graph.initializer.append(numpy_helper.from_array(
            np.ones((args.batch, args.seq), dtype=np.int64), name=inp.name))
        print(f"  baked {inp.name} = ones[{args.batch},{args.seq}]")
    else:
        keep.append(inp)
        print(f"  input {inp.name}: {[d.dim_value for d in dims]}")
del m.graph.input[:]
m.graph.input.extend(keep)

# Bake the attention scale head_dim^-0.25 as a deterministic float32 constant (this is the real
# arm64-vs-x86 divergence: the original folds it via a Shape/Sqrt subgraph that goes to inf on x86).
# The now-dead Sqrt/Shape subgraph is removed by gs.cleanup() below.
import re
scale = float(args.head_dim) ** -0.25
mul_re = re.compile(r"/transformer/h\.\d+/attn/Mul(_\d+)?_output_0$")
scale_names = {n.input[1] for n in m.graph.node
               if n.op_type == "Mul" and n.output and mul_re.search(n.output[0]) and len(n.input) == 2}
kept = [n for n in m.graph.node if not (n.output and n.output[0] in scale_names)]
del m.graph.node[:]
m.graph.node.extend(kept)
for s in scale_names:
    m.graph.initializer.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name=s))
print(f"  baked {len(scale_names)} attention-scale Muls = {args.head_dim}**-0.25 = {scale!r}")

# PURE constant fold (structure-preserving).  onnx-graphsurgeon uses onnxruntime only to
# *evaluate* constant subgraphs, never to fuse/eliminate ops -- so the quant prover's per-matmul
# requant placement stays exactly as in the original model.  (ORT_ENABLE_BASIC's MatMul+Add->Gemm /
# Cast-elimination fusions shifted the requant and overflowed the 2^21 range-check at seq=512.)
import onnx_graphsurgeon as gs
graph = gs.import_onnx(m)
graph.fold_constants(fold_shapes=True)
graph.cleanup().toposort()
out = gs.export_onnx(graph)
onnx.save(out, args.outp)
c = Counter(n.op_type for n in out.graph.node)
print(f"folded -> {len(out.graph.node)} nodes")
shape_ops = {"Shape", "Gather", "Slice", "Reshape", "Concat", "Unsqueeze", "Squeeze",
             "ConstantOfShape", "Range", "Expand"}
remaining = {op: c[op] for op in shape_ops if c.get(op)}
print(f"remaining shape-arithmetic ops: {remaining or 'NONE'}")
# flag any non-standard (contrib) op domains the C++ loader won't know
domains = {n.domain for n in out.graph.node if n.domain not in ("", "ai.onnx")}
print(f"non-standard op domains: {domains or 'none (all standard ONNX)'}")
print(f"op histogram: {dict(c)}")
print(f"wrote {args.outp}")
