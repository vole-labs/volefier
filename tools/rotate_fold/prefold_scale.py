#!/usr/bin/env python3
"""Bake the attention scale (head_dim^-0.25) into the GPT-2 ONNX as a constant.

The HF export traces the 1/sqrt(head_dim) scale as a dynamic
Shape->Gather->...->Sqrt subgraph.  The C++ constant-folder evaluates that
subgraph with arch/build-dependent undefined behaviour (head_dim folds to 0 on
some builds -> scale=inf, 64 on others -> scale=0.354), which makes the proven
circuit differ across machines.

head_dim is a fixed model property (64 for GPT-2), and 64**-0.25 is the same
IEEE double on every platform, so we replace each attention-scale Mul's scale
input with a constant initializer and dead-code-eliminate the now-unused
subgraph.  Output is byte-identical regardless of where this script runs.
"""
import argparse, re, numpy as np, onnx
from onnx import helper, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("outp")
ap.add_argument("--head-dim", type=int, default=64)
args = ap.parse_args()

m = onnx.load(args.inp)
g = m.graph
scale = float(args.head_dim) ** -0.25
print(f"baking attention scale = head_dim^-0.25 = {args.head_dim}**-0.25 = {scale!r}")

# attention-scale Muls: /transformer/h.<i>/attn/Mul(_1)?_output_0 ; 2nd input is the scale
mul_re = re.compile(r"/transformer/h\.\d+/attn/Mul(_\d+)?_output_0$")
scale_names = set()
for n in g.node:
    if n.op_type == "Mul" and n.output and mul_re.search(n.output[0]) and len(n.input) == 2:
        scale_names.add(n.input[1])
print(f"found {len(scale_names)} attention-scale Mul inputs to bake")

# drop any node that *produces* a scale name (the Sqrt), then add a constant initializer
g_nodes = [n for n in g.node if not (n.output and n.output[0] in scale_names)]
del g.node[:]
g.node.extend(g_nodes)
existing_init = {i.name for i in g.initializer}
for s in scale_names:
    if s in existing_init:
        continue
    g.initializer.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name=s))

# iterative dead-code elimination: drop nodes whose outputs feed nothing reachable
graph_outputs = {o.name for o in g.output}
init_names = {i.name for i in g.initializer}
while True:
    consumed = set(graph_outputs)
    for n in g.node:
        consumed.update(n.input)
    dead = [n for n in g.node
            if n.output and all(o not in consumed for o in n.output) and not set(n.output) & graph_outputs]
    if not dead:
        break
    dead_ids = {id(n) for n in dead}
    keep = [n for n in g.node if id(n) not in dead_ids]
    del g.node[:]
    g.node.extend(keep)
print(f"nodes after bake + DCE: {len(g.node)}")

onnx.checker.check_model(m) if m.ByteSize() < 2**31 else print("(skipping checker: >2GB)")
onnx.save(m, args.outp)

# verify: no Sqrt/Shape/Gather left feeding the attention Muls
leftover = [n.op_type for n in g.node
            if "attn" in (n.output[0] if n.output else "") and n.op_type in ("Sqrt", "Slice")]
print(f"leftover attn Sqrt/Slice nodes: {len(leftover)} (want 0)")
print(f"wrote {args.outp}")
