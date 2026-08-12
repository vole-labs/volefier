# Export a QuaRot-rotated GPT-2 ONNX the prover can load directly (residual rotation Q folded into the
# weights; LayerNorms made affine-free; LM head un-shared from the embedding and rotated).  The rotated
# activations are outlier-free, so the prover's STATIC per-tensor int12 path (CALIB) recovers ~fp32.
#   PYTHONPATH=. python3 tools/export_quarot_onnx.py model/gpt2-12_nocache/model.onnx model/gpt2-12-quarot.onnx
import sys, os, numpy as np, onnx
NO_H = not bool(os.environ.get('HADAMARD'))   # DEFAULT: Q-only (rotation-only) export -- ~same accuracy on GPT-2.
                                              # set HADAMARD=1 to also bake the online MLP rotation (higher accuracy).
                                              # The [dff,Nm] weight is a padded Sylvester Hadamard, so the prover runs
                                              # it as a fast Walsh-Hadamard transform (quant_graph_infer.h fast_hadamard,
                                              # RECT mode) -- committed x public, no longer the old dense-matmul cost.
from onnx import numpy_helper as nh, helper, TensorProto
from tools.rotate_fold.quarot_gpt2 import Q_fix_ones, Had

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
W = {t.name: nh.to_array(t).astype(np.float64) for t in g.initializer}
d = W['transformer.wte.weight'].shape[1]; L = 12
Q = Q_fix_ones(d, 0)                                   # ones-fixing rotation: LN0(x@Q)=LN0(x)@Q
dff = W['transformer.h.0.mlp.c_proj.weight'].shape[0]  # MLP intermediate (3072)
Hm, Nm = Had(dff)                                      # online Hadamard, padded to Nm (4096); H[:dff,:] used (gelu is dff-wide)
Hsub = Hm[:dff, :]                                     # [dff, Nm] public online-Hadamard weight (no Pad node needed)

R = {}
R['transformer.wte.weight'] = W['transformer.wte.weight'] @ Q     # rotated embedding (Gather uses this)
R['transformer.wpe.weight'] = W['transformer.wpe.weight'] @ Q
for l in range(L):
    p = f'transformer.h.{l}.'
    for ln, lin in [('ln_1', 'attn.c_attn'), ('ln_2', 'mlp.c_fc')]:
        ga = W[p+ln+'.weight']; be = W[p+ln+'.bias']; Wt = W[p+lin+'.weight']; bt = W[p+lin+'.bias']
        R[p+lin+'.bias']   = bt + be @ Wt                # absorb beta (original W)
        R[p+lin+'.weight'] = Q.T @ (ga[:, None] * Wt)    # absorb gamma into rows, rotate input by Q^T
        R[p+ln+'.weight']  = np.ones_like(ga); R[p+ln+'.bias'] = np.zeros_like(be)   # LN -> affine-free
    R[p+'attn.c_proj.weight'] = W[p+'attn.c_proj.weight'] @ Q     # output -> rotated residual
    R[p+'attn.c_proj.bias']   = W[p+'attn.c_proj.bias'] @ Q
    if NO_H:
        R[p+'mlp.c_proj.weight'] = W[p+'mlp.c_proj.weight'] @ Q    # Q-only: rotate output, keep 3072 input (no online Hadamard)
    else:
        Wpp = np.zeros((Nm, d)); Wpp[:dff] = W[p+'mlp.c_proj.weight']  # pad c_proj input dim to Nm
        R[p+'mlp.c_proj.weight']  = (Hm.T @ Wpp) @ Q                  # absorb online Hadamard (input) + rotation Q (output)
    R[p+'mlp.c_proj.bias']    = W[p+'mlp.c_proj.bias'] @ Q
gf = W['transformer.ln_f.weight']; bf = W['transformer.ln_f.bias']
R['transformer.ln_f.weight'] = np.ones_like(gf); R['transformer.ln_f.bias'] = np.zeros_like(bf)   # ln_f affine-free
lm_w = Q.T @ (gf[:, None] * W['transformer.wte.weight'].T)        # [d, vocab] un-shared rotated LM head
lm_b = bf @ W['transformer.wte.weight'].T                         # ln_f beta -> LM bias

# rewrite initializers
keep = []
for t in g.initializer:
    keep.append(nh.from_array(R[t.name].astype(np.float32), t.name) if t.name in R else t)
del g.initializer[:]; g.initializer.extend(keep)
g.initializer.append(nh.from_array(lm_w.astype(np.float32), 'lm_head_quarot'))
g.initializer.append(nh.from_array(lm_b.astype(np.float32), 'lm_bias_quarot'))

# repoint the LM-head MatMul to the un-shared weight, and add the bias as an Add producing 'logits'
for n in g.node:
    if n.op_type == 'MatMul' and n.output[0] == 'logits':
        n.input[1] = 'lm_head_quarot'; n.output[0] = 'logits_pre'
        g.node.append(helper.make_node('Add', ['logits_pre', 'lm_bias_quarot'], ['logits']))
        break

# insert the online MLP Hadamard (public, name marks it) before each mlp.c_proj  (skipped in Q-only mode)
hl = 0
if not NO_H:
    g.initializer.append(nh.from_array(Hsub.astype(np.float32), 'mlp_hadamard'))
    nn = []
    for n in g.node:
        if n.op_type in ('MatMul', 'Gemm') and len(n.input) > 1 and n.input[1].endswith('mlp.c_proj.weight'):
            ho = f'mlp_had_{hl}'; hl += 1
            nn.append(helper.make_node('MatMul', [n.input[0], 'mlp_hadamard'], [ho])); n.input[0] = ho
        nn.append(n)
    del g.node[:]; g.node.extend(nn)
onnx.save(m, dst)
print(f'wrote {dst}: rotated {len(R)} tensors, un-shared LM head [{lm_w.shape}], {hl} MLP Hadamards [{Hsub.shape}], ln affine-free')
