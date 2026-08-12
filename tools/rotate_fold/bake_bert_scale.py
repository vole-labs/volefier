#!/usr/bin/env python3
# BERT twin of tools/prefold_scale.py: the HF BERT export traces the attention scale
# (Shape->Gather->Cast->Sqrt->Div->...) live in the graph; ml-quant's ref for that chain
# is garbage (MIMIC shows +-5e17 at attention/self/Mul) -> mult-check fails.
# Bake head_dim^-0.25 as a constant into each attention-scale Mul input, then DCE the chain.
import sys, onnx, numpy as np
from onnx import numpy_helper, helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
init = {t.name for t in g.initializer}

def chain_has_sqrt(name, depth=0):
    if depth > 12 or name in init: return False
    n = prod.get(name)
    if n is None: return False
    if n.op_type == "Sqrt": return True
    return any(chain_has_sqrt(i, depth + 1) for i in n.input)

SCALE = np.float32(64 ** -0.25)   # bert-base head_dim=64; each of Q,K scaled by d^-0.25
baked = 0
for n in g.node:
    if n.op_type != "Mul" or "attention/self/Mul" not in n.name: continue
    for idx, inp in enumerate(n.input):
        if inp not in init and chain_has_sqrt(inp):
            cname = f"baked_attn_scale_{baked}"
            g.initializer.append(numpy_helper.from_array(np.array(SCALE, dtype=np.float32), cname))
            n.input[idx] = cname
            baked += 1
print(f"baked {baked} attention-scale Mul inputs to {SCALE}")

# DCE: iteratively drop nodes whose outputs feed neither another node nor a graph output
# (removes the whole orphaned Shape->Gather->Cast->Sqrt->Div scale chain).
dropped = 0
changed = True
while changed:
    changed = False
    consumed = set(o.name for o in g.output)
    for n in g.node:
        for i in n.input: consumed.add(i)
    newnodes = [n for n in g.node if any(o in consumed for o in n.output)]
    if len(newnodes) != len(g.node):
        dropped += len(g.node) - len(newnodes)
        del g.node[:]
        g.node.extend(newnodes)
        changed = True
print(f"DCE dropped {dropped} nodes; remaining {len(g.node)}")
ops = {}
for n in g.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("Sqrt remaining:", ops.get("Sqrt", 0), "| Div:", ops.get("Div", 0))
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst)
