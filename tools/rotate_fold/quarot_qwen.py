# QuaRot-style residual rotation for Qwen2.5 (plaintext validation, BEFORE porting to the prover).
# Qwen uses RMSNorm (no mean-subtraction) -> RMSNorm(x@Q)=RMSNorm(x)@Q for ANY orthonormal Q, so unlike
# GPT-2's LayerNorm we need NO "Q fixes 1" constraint.  We:
#   (a) absorb each RMSNorm weight gamma into the FOLLOWING linear(s) so the norm is affine-free;
#   (b) rotate the residual stream by one orthonormal Q, folded into weights (FREE at proving):
#         input side  (read residual):  W <- (W @ diag(gamma)) @ Q   [embed, q/k/v/gate/up, lm_head]
#         output side (write residual): W <- Q^T @ W                 [o_proj, down_proj]
# This smooths the residual-stream / RMSNorm-input outliers so a single per-tensor (STATIC) scale
# suffices -- removing the per-token scale REVEAL that dynamic calibration needs.
# (R2 attention-value / R4 SwiGLU-intermediate online Hadamards are separate; this validates R1 first.)
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from transformers import AutoModelForCausalLM

hf = "Qwen/Qwen2.5-0.5B-Instruct"
if len(sys.argv) > 1 and sys.argv[1] != "export":
    hf = sys.argv[1]
elif len(sys.argv) > 4 and sys.argv[1] == "export":   # export <dir> <seq> <hf_id> (else defaults to Qwen)
    hf = sys.argv[4]
ids = [785,3974,13876,38835,34208,916,279,15678,5562,1393,279,7015,7289,13970,4815,279]

def load():
    return AutoModelForCausalLM.from_pretrained(hf, use_cache=False, attn_implementation="eager", dtype=torch.float32).eval()

def rand_orthonormal(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q.to(torch.float32)

def sylvester(n):  # n = 2^k, +/-1 Sylvester-Hadamard, orthonormal (/sqrt n); fast WHT applies (O(n log n))
    H = np.array([[1.0]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def structured_hadamard(d, seed=1):
    # SAME-DIM structured Hadamard (no padding): d = 2^k * m -> H = O_m (kron) H_{2^k} .
    # Reshape x as [m, 2^k]: WHT (fast, O(2^k log 2^k)) along the 2^k axis, tiny dense m x m block
    # along the m axis (m odd -> no +/-1 Hadamard exists). kron of orthonormals is orthonormal.
    # Qwen I=4864=256*19 -> fast WHT on the 256 factor, dense 361-entry 19x19 on the rest.
    # Validated: R1+R4 with this H recovers 16/16 (== dense random) at static int12; fast-WHT-able so
    # the prover never materializes the 4864^2 dense matrix (kills the ~2.3GB constant + memory blowup).
    p2 = d & (-d)                                    # largest power of 2 dividing d (4864 -> 256)
    Om = rand_orthonormal(d // p2, seed).numpy().astype(np.float64)  # 19x19
    return torch.from_numpy(np.kron(Om, sylvester(p2))).to(torch.float32)

@torch.no_grad()
def apply_quarot(m, Q):
    md = m.model
    if md.embed_tokens.weight.data_ptr() == m.lm_head.weight.data_ptr():
        m.lm_head.weight = torch.nn.Parameter(md.embed_tokens.weight.data.clone())  # UNTIE: embed & head rotate differently
    md.embed_tokens.weight.copy_(md.embed_tokens.weight @ Q)   # embed writes residual -> rotate output cols
    for lyr in md.layers:
        g1 = lyr.input_layernorm.weight.clone()
        for lin in (lyr.self_attn.q_proj, lyr.self_attn.k_proj, lyr.self_attn.v_proj):
            lin.weight.copy_((lin.weight * g1) @ Q)            # absorb gamma (input cols) then rotate input
        lyr.input_layernorm.weight.fill_(1.0)
        g2 = lyr.post_attention_layernorm.weight.clone()
        for lin in (lyr.mlp.gate_proj, lyr.mlp.up_proj):
            lin.weight.copy_((lin.weight * g2) @ Q)
        lyr.post_attention_layernorm.weight.fill_(1.0)
        lyr.self_attn.o_proj.weight.copy_(Q.t() @ lyr.self_attn.o_proj.weight)  # writes residual -> rotate output rows
        lyr.mlp.down_proj.weight.copy_(Q.t() @ lyr.mlp.down_proj.weight)
    gf = md.norm.weight.clone()
    m.lm_head.weight.copy_((m.lm_head.weight * gf) @ Q)         # reads final norm out -> input projection
    md.norm.weight.fill_(1.0)
    return m

@torch.no_grad()
def apply_r4(m, Hd):   # R4: online Hadamard on the SwiGLU intermediate (down_proj input), folded into down_proj
    for lyr in m.model.layers:
        lyr.mlp.down_proj.weight.copy_(lyr.mlp.down_proj.weight @ Hd)   # down reads z@H -> W <- W @ H
    return m

@torch.no_grad()
def apply_r2(m, Hh):   # R2: per-head Hadamard on value/context head_dim -> folds into v_proj rows + o_proj cols
    cfg = m.config; dh = cfg.hidden_size // cfg.num_attention_heads
    nkv, nq = cfg.num_key_value_heads, cfg.num_attention_heads
    for lyr in m.model.layers:
        v, o = lyr.self_attn.v_proj, lyr.self_attn.o_proj
        v.weight.copy_((Hh.t() @ v.weight.view(nkv, dh, -1)).reshape(nkv * dh, -1))   # V_kv <- H^T @ V_kv (rows)
        if v.bias is not None: v.bias.copy_((v.bias.view(nkv, dh) @ Hh).reshape(-1))
        o.weight.copy_((o.weight.view(-1, nq, dh) @ Hh).reshape(o.weight.shape[0], nq * dh))  # un-rotate cols
    return m

def fakequant_hooks(m, bits=12, Hd=None, quant_v=False):
    LIM = (1 << (bits - 1)) - 1; hs = []
    def q(x): s = x.abs().amax() / LIM + 1e-30; return torch.clamp(torch.round(x / s), -LIM, LIM) * s  # per-tensor static
    def plain(mod, inp): return (q(inp[0]),)
    def had(mod, inp): return (q(inp[0] @ Hd),)                # R4: rotate down-input by H, THEN quantize (smoothed)
    def vout(mod, inp, out): return q(out)                     # quantize the value-projection OUTPUT (R2 smooths this)
    for lyr in m.model.layers:
        for lin in (lyr.self_attn.q_proj, lyr.self_attn.k_proj, lyr.self_attn.v_proj, lyr.self_attn.o_proj,
                    lyr.mlp.gate_proj, lyr.mlp.up_proj):
            hs.append(lin.register_forward_pre_hook(plain))
        hs.append(lyr.mlp.down_proj.register_forward_pre_hook(had if Hd is not None else plain))
        if quant_v: hs.append(lyr.self_attn.v_proj.register_forward_hook(vout))
    return hs

@torch.no_grad()
def argmax(m):
    return m(input_ids=torch.tensor([ids])).logits[0].argmax(-1)

@torch.no_grad()
def export_rotated(outdir, seq):
    import torch.nn as nn
    m = load(); H = m.config.hidden_size; I = m.config.intermediate_size
    Q = rand_orthonormal(H, 0)
    Hd = structured_hadamard(I, 1)                            # STRUCTURED (kron O_m,H_p2) -> prover fast-WHT path
    apply_quarot(m, Q); apply_r4(m, Hd)                       # R1 folded; R4 Hd folded into down_proj weight
    for lyr in m.model.layers:                                # insert ONLINE Hadamard z@Hd before down_proj
        had = nn.Linear(I, I, bias=False); had.weight.copy_(Hd.t())   # had(z) = z @ Hd
        lyr.mlp.down_proj = nn.Sequential(had, lyr.mlp.down_proj)
    m.eval(); os.makedirs(outdir, exist_ok=True)
    class Wr(nn.Module):
        def __init__(s, mm): super().__init__(); s.m = mm
        def forward(s, input_ids, attention_mask, position_ids):
            return s.m(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits
    v = m.config.vocab_size
    a = (torch.randint(0, v, (1, seq)), torch.ones(1, seq, dtype=torch.long), torch.arange(seq).unsqueeze(0))
    path = os.path.join(outdir, "model.onnx")
    torch.onnx.export(Wr(m), a, path, input_names=["input_ids", "attention_mask", "position_ids"],
                      output_names=["logits"], dynamic_axes=None, opset_version=18, do_constant_folding=True, dynamo=False)
    print("wrote rotated (R1+R4) ONNX:", path, os.path.getsize(path) / 1e6, "MB (+ external)")

if len(sys.argv) > 1 and sys.argv[1] == "export":
    export_rotated(sys.argv[2] if len(sys.argv) > 2 else "model/qwen2.5-0.5b-quarot",
                   int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    sys.exit(0)

print("loading fp32 reference ...")
base = load(); fp = argmax(base)
Q = rand_orthonormal(base.config.hidden_size, 0)

print("checking rotation EXACTNESS (rotated fp32 == original) ...")
rot = apply_quarot(load(), Q); rl = argmax(rot)
exact = int((fp == rl).sum())
print(f"  exact-match argmax: {exact}/16   (should be 16/16 -- rotation is math-equivalent)")

Hd = rand_orthonormal(base.config.intermediate_size, 1)                          # R4: SwiGLU intermediate
dh = base.config.hidden_size // base.config.num_attention_heads
Hh = rand_orthonormal(dh, 2)                                                      # R2: per-head value/context
print("static per-tensor int12 fake-quant (linear inputs), configs:")
m0 = load(); fakequant_hooks(m0, 12); a0 = int((argmax(m0) == fp).sum())
m1 = apply_quarot(load(), Q); fakequant_hooks(m1, 12); a1 = int((argmax(m1) == fp).sum())
m2 = apply_r4(apply_quarot(load(), Q), Hd); fakequant_hooks(m2, 12, Hd); a2 = int((argmax(m2) == fp).sum())
Hs = structured_hadamard(base.config.intermediate_size, 1)                        # R4 STRUCTURED (fast-WHT-able)
orth = float((Hs @ Hs.t() - torch.eye(Hs.shape[0])).abs().max())                  # sanity: orthonormal?
m2s = apply_r4(apply_quarot(load(), Q), Hs); fakequant_hooks(m2s, 12, Hs); a2s = int((argmax(m2s) == fp).sum())
print(f"  no-rotation     static int12: {a0}/16")
print(f"  QuaRot R1       static int12: {a1}/16")
print(f"  QuaRot R1+R4    static int12: {a2}/16   (R4 = dense RANDOM-orthonormal Hadamard)")
print(f"  QuaRot R1+R4*   static int12: {a2s}/16   (R4* = STRUCTURED H_256(x)O_19, ||HH^T-I||={orth:.1e}, fast-WHT-able)")
print("\nnow ALSO quantizing the value-projection output (attention region the ZK path hits):")
m3 = apply_r4(apply_quarot(load(), Q), Hd); fakequant_hooks(m3, 12, Hd, quant_v=True); a3 = int((argmax(m3) == fp).sum())
m4 = apply_r2(apply_r4(apply_quarot(load(), Q), Hd), Hh); fakequant_hooks(m4, 12, Hd, quant_v=True); a4 = int((argmax(m4) == fp).sum())
print(f"  QuaRot R1+R4      + V-quant: {a3}/16   (no R2 -> value outliers hurt)")
print(f"  QuaRot R1+R4+R2   + V-quant: {a4}/16   (R2 = per-head value/context Hadamard)")
