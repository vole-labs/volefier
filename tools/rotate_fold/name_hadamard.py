# Rename the shared online-Hadamard weight so the prover's fast-WHT handler detects it
# (quant_graph_infer.h keys on the substring "hadamard").  Edits only the graph proto (the small
# .onnx); the external .data (weights) is untouched.  Idempotent.
import sys, onnx
path = sys.argv[1] if len(sys.argv) > 1 else "model/qwen2.5-0.5b-quarot-fused.onnx"
m = onnx.load(path, load_external_data=False); g = m.graph
# online-Hadamard MatMuls: node name contains "down_proj.0" (the nn.Sequential Hadamard Linear)
targets = {}
for n in g.node:
    if n.op_type == "MatMul" and "down_proj.0" in n.name and len(n.input) >= 2:
        w = n.input[1]
        if "hadamard" not in w: targets.setdefault(w, "mlp_hadamard_" + w.split("_")[-1])
if not targets:
    print("no online-Hadamard weights to rename (already named or none found)"); sys.exit(0)
for i in g.initializer:
    if i.name in targets: i.name = targets[i.name]
for n in g.node:
    for k, inp in enumerate(n.input):
        if inp in targets: n.input[k] = targets[inp]
# Write ONLY the graph proto (renamed initializer name + node input refs). External-data
# locations/offsets in each TensorProto are unchanged by the rename, so the .data file is left
# byte-for-byte intact. (onnx.save() here would re-serialize and APPEND a full copy of the external
# data, ~doubling the .data -- avoid it.)
with open(path, "wb") as f:
    f.write(m.SerializeToString())
print("renamed:", targets)