# Fuse decomposed LLM subgraphs back into single ops the ml-quant / ml-fixed front end can
# dispatch to its gadgets (QuantRMSNorm, build_silu_table, ...). torch.onnx.export lowers these
# to primitive chains; we pattern-match and collapse them. Run on the FIXED-shape folded export
# (FIXED_SEQ=<n> + ORT fold), which first removes the dynamic-shape/mask junk.
#
#   RMSNorm:  Pow(x,2) -> ReduceMean -> Add(eps) -> Sqrt -> Div(x,.) -> Mul(weight)
#                 =>  RMSNormalization(x, weight)   [attr: epsilon, axis=-1]
#   SiLU:     Sigmoid(x) ; Mul(x, sigmoid)          =>  SiLU(x)
#
# Usage:  python tools/fuse_llm_onnx.py <in.onnx> <out.onnx>
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.shape_inference import quant_pre_process

src = sys.argv[1] if len(sys.argv) > 1 else "model/qwen2.5-0.5b/model.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "model/qwen2.5-0.5b-fused.onnx"

# 1) bake position_ids / attention_mask (fixed-seq graph) to constants, then constant-fold, so the
#    rotary Cos/Sin tables and the causal mask collapse to initializers (prereq for RoPE fusion).
pre = onnx.load(src, load_external_data=True); pg = pre.graph
dim = lambda vi: [d.dim_value for d in vi.type.tensor_type.shape.dim]
shp = {i.name: dim(i) for i in pg.input}
if shp.get("input_ids", [0, 0])[1] in (0, None):
    sys.exit("fuse_llm_onnx expects a FIXED-shape export (FIXED_SEQ=<n> tools/export_llm_onnx.py); "
             "input_ids sequence dim is dynamic.")
S = shp["input_ids"][1]
bake = {"position_ids": np.arange(S, dtype=np.int64)[None, :], "attention_mask": np.ones((1, S), np.int64)}
keep = [i for i in pg.input if i.name not in bake]
del pg.input[:]; pg.input.extend(keep)
for nm, arr in bake.items():
    if nm not in {i.name for i in pg.initializer}: pg.initializer.append(numpy_helper.from_array(arr, nm))
baked = dst.replace(".onnx", "-baked.onnx")
onnx.save(pre, baked, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(baked) + ".data")
folded = dst.replace(".onnx", "-folded.onnx")
quant_pre_process(baked, folded, skip_symbolic_shape=False,
                  save_as_external_data=True, all_tensors_to_one_file=True)
print(f"baked position/mask (seq={S}) and folded -> now fusing")

m = onnx.load(folded)                               # loads external weights too (re-saved below)
g = m.graph
inits = {i.name: i for i in g.initializer}
cons = {}
prod = {}
for n in g.node:
    for o in n.output: prod[o] = n
    for i in n.input: cons.setdefault(i, []).append(n)

def only(t):                                         # unique consumer of tensor t, else None
    c = cons.get(t, []); return c[0] if len(c) == 1 else None
def val(name):
    return numpy_helper.to_array(inits[name]) if name in inits else None

drop = set()            # node names to delete
repl = {}               # chain-head node name -> fused node (inserted in its place)

# ---- RMSNorm ----
rms = 0
for n in g.node:
    if n.op_type != "Pow": continue
    e = val(n.input[1])
    if e is None or float(np.ravel(e)[0]) != 2.0: continue
    x = n.input[0]
    rm = only(n.output[0]);                       # ReduceMean
    if not rm or rm.op_type != "ReduceMean": continue
    ad = only(rm.output[0])                        # Add(eps)
    if not ad or ad.op_type != "Add": continue
    eps_in = ad.input[1] if ad.input[0] == rm.output[0] else ad.input[0]
    sq = only(ad.output[0])                        # Sqrt
    if not sq or sq.op_type != "Sqrt": continue
    dv = only(sq.output[0])                        # Div(x, std)
    if not dv or dv.op_type != "Div" or dv.input[0] != x or dv.input[1] != sq.output[0]: continue
    ml = only(dv.output[0])                        # Mul(weight, .)
    if not ml or ml.op_type != "Mul": continue
    w = ml.input[0] if ml.input[1] == dv.output[0] else ml.input[1]
    if w not in inits: continue
    ev = val(eps_in); eps = float(np.ravel(ev)[0]) if ev is not None else 1e-6
    repl[n.name] = helper.make_node("RMSNormalization", [x, w], [ml.output[0]],
                                    name=n.name + "_rms", epsilon=eps, axis=-1)
    for nd in (n, rm, ad, sq, dv, ml): drop.add(nd.name)
    rms += 1

# ---- SiLU:  x * sigmoid(x) ----
silu = 0
for n in g.node:
    if n.op_type != "Sigmoid" or n.name in drop: continue
    x = n.input[0]
    for c in cons.get(n.output[0], []):
        if c.op_type == "Mul" and c.name not in drop and x in c.input and n.output[0] in c.input:
            repl[n.name] = helper.make_node("SiLU", [x], [c.output[0]], name=n.name + "_silu")
            drop.add(n.name); drop.add(c.name); silu += 1
            break

# ---- RoPE:  Add( Mul(x, cos), Mul( Concat(Neg(Slice(x,2nd)), Slice(x,1st)), sin ) ) ----
# cos/sin must be constants (bake position_ids/attention_mask to initializers + fold first).
rope = 0
for a in g.node:
    if a.op_type != "Add" or a.name in drop: continue
    p0, p1 = prod.get(a.input[0]), prod.get(a.input[1])
    if not p0 or not p1 or p0.op_type != "Mul" or p1.op_type != "Mul": continue
    hit = None
    for mc, ms in ((p0, p1), (p1, p0)):              # mc = Mul(x,cos), ms = Mul(rot,sin)
        concat = next((prod.get(i) for i in ms.input if prod.get(i) and prod.get(i).op_type == "Concat"), None)
        if not concat: continue
        sin = next((i for i in ms.input if not (prod.get(i) and prod.get(i).op_type == "Concat")), None)
        cps = [prod.get(i) for i in concat.input]
        neg = next((c for c in cps if c and c.op_type == "Neg"), None)
        slc1 = next((c for c in cps if c and c.op_type == "Slice"), None)
        if not neg or not slc1: continue
        slc2 = prod.get(neg.input[0])
        if not slc2 or slc2.op_type != "Slice": continue
        x = slc1.input[0]
        if slc2.input[0] != x or x not in mc.input: continue
        cos = next((i for i in mc.input if i != x), None)
        if cos not in inits or sin not in inits: continue          # require constant cos/sin
        hit = (x, cos, sin, [mc.name, ms.name, concat.name, neg.name, slc1.name, slc2.name])
        break
    if not hit: continue
    x, cos, sin, sub = hit
    repl[a.name] = helper.make_node("RotaryEmbedding", [x, cos, sin], [a.output[0]], name=a.name + "_rope")
    for nm in sub: drop.add(nm)
    rope += 1

out = []
for n in g.node:
    if n.name in repl: out.append(repl[n.name])
    elif n.name in drop: continue
    else: out.append(n)
del g.node[:]; g.node.extend(out)

onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(dst) + ".data")
for f in (baked, baked + ".data", folded, folded + ".data"):   # drop intermediates
    if os.path.exists(f): os.remove(f)
print(f"fused RMSNorm x{rms}, SiLU x{silu}, RoPE x{rope}; graph now {len(g.node)} nodes -> {dst}")
