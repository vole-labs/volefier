# SCHEME-exact plaintext mimic of the ml-quant GRAPH prover on gpt2-folded-s512 (data-movement skip).
# Reproduces every gadget: per-tensor int12 (dynamic self-calibration), scale_search c*2^e round-half-up
# requant, sqrt-free int LayerNorm, exp-LUT LSE softmax (FUSED_NL), gelu LUT, data-movement transparency.
# FAST: int matmul accumulators fit float64's 53-bit mantissa, so we use float64 BLAS + round -> exact ints
# (seconds, vs ~13 min for an int64 forward).
#
# FIDELITY: gadget-for-gadget identical to the prover; leading tokens match exactly (12/12) and ~84% over
# 512.  It is NOT bit-exact over a long sequence: it self-calibrates per-tensor scales dynamically rather
# than using the prover's FROZEN c512_skip.calib scales, and it collapses the attention graph
# (MatMul->Mul->Mul_1->Where->Softmax->MatMul_1) into single steps.  For a bit-exact reference use the
# prover's own internal ref_ (exact by construction) or a graph-driven interpreter of the folded ONNX.
#
#   PYTHONPATH=. python3 tools/exact_mimic_folded.py [zk_tokens_file]
import sys, os, numpy as np
from tools.rotate_fold.quarot_gpt2 import load

P = 12; L = 2047; SP = 1 << P
GT = 1 << (P + 8); GOFF = 1 << (P + 7); ET = 1 << (P + 9)
_xx = (np.arange(GT) - GOFF) / SP
GEL = np.round(0.5 * _xx * (1 + np.tanh(np.sqrt(2/np.pi) * (_xx + 0.044715 * _xx**3))) * SP).astype(np.int64)
EXP_NC = np.round(np.exp(-np.arange(ET) / SP) * SP).astype(np.int64)   # no-clamp (FUSED_NL LSE): tail/masked -> 0

def imm(A, B): return np.rint(A.astype(np.float64) @ B.astype(np.float64)).astype(np.int64)  # exact int matmul
def scale_search(target):
    best = 1e300; bc = 1; be = 0
    for e in range(-24, 11):
        c0 = int(round(target / (2.0**e)))
        for c in range(max(1, c0 - 1), min(800, c0 + 1) + 1):
            d = abs(c * (2.0**e) - target)
            if d < best: best, bc, be = d, c, e
    return bc, be
def rqu(acc, c, e):
    acc = np.asarray(acc, np.int64)
    if e >= 0: return acc * c * (1 << e)
    M = -e; N = acc * c
    return (N + (np.int64(1) << (M - 1))) >> M
def REQ(acc, ratio): c, e = scale_search(ratio); return rqu(acc, c, e)
def rqP(N, M): N = np.asarray(N, np.int64); return (N + (np.int64(1) << (M - 1))) >> M
def rdiv(a, D):
    a = np.asarray(a, np.int64); h = D // 2
    return np.where(a >= 0, (a + h) // D, -(((-a) + h) // D))
def qpt(X, s):                          # per-tensor int12 (dynamic max == static calib on the calib input)
    so = max(float(np.abs(X).max()) * s / L, 1e-30); return REQ(X, s / so), so
def wq(Wfp):
    sw = max(float(np.abs(Wfp).max()) / L, 1e-30); return np.clip(np.round(Wfp / sw), -L, L).astype(np.int64), sw
def mm(X, sx, Wfp, bfp=None, to_pt=True):
    q, sw = wq(Wfp); acc = imm(X, q); sacc = sx * sw
    if bfp is not None: acc = acc + np.round(bfp / sacc).astype(np.int64)
    return qpt(acc, sacc) if to_pt else (acc, sacc)
def gelu(X, sx):
    xp = REQ(X, sx * SP); g = GEL[np.clip(xp + GOFF, 0, GT - 1)]; return qpt(g, 1.0 / SP)
def lnorm(X, sx, d):                    # sqrt-free int LayerNorm (affine absorbed -> LN0)
    mu = rdiv(X.sum(-1, keepdims=True), d); c = X - mu
    vq = rdiv((c.astype(np.int64)**2).sum(-1, keepdims=True), d) + 1
    dinv = np.round((1 << (2*P)) / np.sqrt(vq)).astype(np.int64)
    return qpt(rqP(c * dinv, P), 1.0 / SP)
def add(A, sa, B, sb): return qpt(A + REQ(B, sb / sa), sa)   # residual: align B->A scale, add, requant

def forward(R, ids, H=12):
    # R = weights loaded DIRECTLY from gpt2-folded-s512.onnx: rotation baked in, LN affine absorbed
    # (gamma=ones/beta=zeros -> LN0), no online Hadamard, lm_head_quarot.
    d = R['transformer.wte.weight'].shape[1]; dh = d // H; S = len(ids)
    x, sx = qpt(np.round((R['transformer.wte.weight'][ids] + R['transformer.wpe.weight'][np.arange(S)]) * SP).astype(np.int64), 1.0/SP)
    cm = np.tril(np.ones((S, S)))
    for l in range(12):
        p = f'transformer.h.{l}.'
        h, sh = lnorm(x, sx, d)
        qkv, sqk = mm(h, sh, R[p+'attn.c_attn.weight'], R[p+'attn.c_attn.bias'])
        q, k, v = [qkv[:, i*d:(i+1)*d].reshape(S, H, dh).transpose(1, 0, 2) for i in range(3)]
        scP = np.where(cm[None] > 0, REQ(imm(q, k.transpose(0, 2, 1)), sqk*sqk*SP/np.sqrt(dh)), -10**9)  # @2^-P
        # LSE softmax (FUSED_NL): out_i = exp_table[(m - sc_i) + shift], shift = round(log(sum_exp)*2^P); no division
        m = scP.max(-1, keepdims=True); t = m - scP
        sumexp = np.exp(-np.clip(t, 0, None) / SP).sum(-1, keepdims=True)
        shift = np.maximum(np.round(np.log(sumexp) * SP).astype(np.int64), 0)
        attn = EXP_NC[np.clip(t + shift, 0, ET - 1)]
        a, sa = qpt(imm(attn, v).transpose(1, 0, 2).reshape(S, d), (1.0/SP)*sqk)
        ao, sao = mm(a, sa, R[p+'attn.c_proj.weight'], R[p+'attn.c_proj.bias'])
        x, sx = add(x, sx, ao, sao)
        h, sh = lnorm(x, sx, d)
        fc, sfc = mm(h, sh, R[p+'mlp.c_fc.weight'], R[p+'mlp.c_fc.bias'])
        go, sg = gelu(fc, sfc)
        mo, smo = mm(go, sg, R[p+'mlp.c_proj.weight'], R[p+'mlp.c_proj.bias'])
        x, sx = add(x, sx, mo, smo)
    h, sh = lnorm(x, sx, d)
    acc, _ = mm(h, sh, R['lm_head_quarot'], R['lm_bias_quarot'], to_pt=False)   # logits (argmax scale-free)
    return acc

if __name__ == '__main__':
    R = load('model/gpt2-folded-s512.onnx')
    ids = [100 + i for i in range(512)]
    am = forward(R, ids).argmax(-1).tolist()
    print("mimic first 12:", am[:12])
    if len(sys.argv) > 1:
        zk = [int(t) for t in open(sys.argv[1]).read().split()]
        n = min(len(zk), len(am)); agree = sum(a == b for a, b in zip(am[:n], zk[:n]))
        print(f"vs ZK prover ({n} tokens): agree {agree}/{n}  ({100*agree/n:.1f}%)")
