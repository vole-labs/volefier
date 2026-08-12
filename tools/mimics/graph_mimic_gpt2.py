# Graph-driven BIT-EXACT plaintext (numpy) mimic of the ml-quant GRAPH prover on
# model/gpt2-folded-s512.onnx, with the DeepProve-style STATIC per-tensor int12 calibration
# (/tmp/c512_skip.calib) + data-movement skip.  Ports the prover's ALICE plaintext `ref_`
# formulas from volefier/ml-quant/{quant_graph_infer.h,quant_infer.h,quant_nonlinear.h}
# VERBATIM (matching every rounding direction and scale source), and walks the real GPT-2
# op graph (multi-op attention with per-op static scales, fused tanh-gelu).
#
# FAST: int matmul accumulators fit float64's 53-bit mantissa, so exact int matmuls use
# float64 BLAS + round.  No ZK proof is run.
#
#   PYTHONPATH=. python3 tools/graph_mimic_gpt2.py [zk_tokens_file]
#
# Rounding conventions ported from the prover:
#   - requant_cm_one / sym_quant_act / rescale_to_P / Mul / LN final rq : ROUND-HALF-UP
#     q = (N + 2^(M-1)) >> M   (arithmetic shift = floor, so half-up toward +inf)
#   - LayerNorm mu / var rdiv : ROUND-HALF-AWAY  (N>=0:(N+D/2)//D ; N<0:-((-N+D/2)//D))
#   - llround (weight quant, bias, dinv, gelu out, softmax exp/shift, embed table) : HALF-AWAY
import sys, os, numpy as np
from tools.rotate_fold.quarot_gpt2 import load

P = 12; L = 2047; SP = 1 << P                    # int12 activations/weights, 2^-P bus
CALIB_PATH = os.environ.get("CALIB", "/tmp/c512_skip.calib")

# ---------- calibration (STATIC per-tensor scales) ----------
CAL = {}
for line in open(CALIB_PATH):
    line = line.split()
    if len(line) == 2:
        CAL[line[0]] = float(line[1])
def cs(name):                                    # static per-tensor scale = calib_max / L
    return max(CAL[name] / L, 1e-30)

# ---------- rounding primitives (VERBATIM from the prover) ----------
def rha(x):                                      # llround: round half away from zero
    x = np.asarray(x, np.float64)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)
def imm(A, B):                                   # exact int matmul (2-D), acc fits in float64
    return np.rint(A.astype(np.float64) @ B.astype(np.float64)).astype(np.int64)
def imm3(A, B):                                  # batched exact int matmul [G,m,k]@[G,k,n]
    return np.rint(np.matmul(A.astype(np.float64), B.astype(np.float64))).astype(np.int64)

_SS = {}
def scale_search(target):                        # zkGPT: best c*2^e, c in [1,800], e in [-24,10]
    r = _SS.get(target)                          # VERBATIM full scan (same loop order + strict <)
    if r is not None: return r
    cc = np.arange(1, 801, dtype=np.float64)
    best = 1e300; bc = 1; be = 0
    for e in range(-24, 11):
        d = np.abs(np.ldexp(cc, e) - target)
        j = int(d.argmin())                      # argmin returns FIRST min -> matches strict-< earliest c
        if d[j] < best: best, bc, be = d[j], j + 1, e
    _SS[target] = (bc, be)
    return bc, be
def rqu(acc, c, e):                              # requant_cm_one core: round-half-up by c*2^e
    acc = np.asarray(acc, np.int64)
    if e >= 0: return acc * c * (1 << e)
    M = -e; N = acc * c
    return (N + (np.int64(1) << (M - 1))) >> M   # arithmetic shift = floor -> half-up
def rqP(N):                                      # round-half-up by 2^P (Mul / LN final)
    N = np.asarray(N, np.int64)
    return (N + (np.int64(1) << (P - 1))) >> P
def rdiv(a, D):                                  # LayerNorm mu/var: round-half-away division
    a = np.asarray(a, np.int64); h = D // 2
    return np.where(a >= 0, (a + h) // D, -(((-a) + h) // D))

def REQ(acc, ratio):                             # scale_search + round-half-up (a requant by a ratio)
    c, e = scale_search(ratio); return rqu(acc, c, e)
def requant_static(acc, s_in, name):             # sym_quant_act with a fixed static calib scale
    s_o = cs(name)
    return REQ(acc, s_in / s_o), s_o

# ---------- weight quant (matmul handler: per-tensor symmetric int12) ----------
def wq(Wfp):
    mxw = max(float(np.abs(Wfp).max()), 1e-12)
    swq = mxw / L
    q = np.clip(rha(Wfp / swq), -L, L).astype(np.int64)
    return q, swq

# ---------- op handlers (each ports the prover's ref_ block) ----------
def matmul(X, sx, Wfp, bfp=None, calib_name=None):
    # Gemm/MatMul ref: acc = beff + X_q @ W_q ; scale = sx*sw ; then static requant to calib.
    qW, sw = wq(Wfp)
    acc = imm(X, qW)
    sacc = sx * sw
    if bfp is not None:
        acc = acc + rha(bfp / sacc)              # beff[o] = llround(breal / (sx*sw)), xz=0
    if calib_name is None:
        return acc, sacc                          # lm head: caller requants to logits_pre
    return requant_static(acc, sacc, calib_name)

def add_resid(A, sa, B, sb, calib_name):
    # Add ref (two activations, zp=0): rescale B -> A's scale (round-half-up), add, requant to calib.
    if sb != sa:
        c, e = scale_search(sb / sa); bq = rqu(B, c, e)
    else:
        bq = np.asarray(B, np.int64)
    y = np.asarray(A, np.int64) + bq
    return requant_static(y, sa, calib_name)

def layernorm(X, d, calib_name, gfp, bfp):
    # QuantLayerNorm ref: sqrt-free int LN on the NATIVE ints, gamma/beta @2^P, out @2^-P; then requant.
    qg = rha(gfp * SP); qb = rha(bfp * SP)
    X = np.asarray(X, np.int64)
    mu = rdiv(X.sum(-1, keepdims=True), d)
    c = X - mu
    vq = rdiv((c * c).sum(-1, keepdims=True), d) + 1
    dinv = rha((1 << (2 * P)) / np.sqrt(vq.astype(np.float64)))
    out = rqP(rqP(c * dinv) * qg) + qb           # rq(rq(c*dinv)*qg)+qb ; scale 2^-P
    return requant_static(out, 1.0 / SP, calib_name)

GT = 1 << (P + 8); GOFF = 1 << (P + 7)            # generic gelu table domain / offset
def gelu(fc_int, s_in, calib_out):
    k = np.sqrt(2.0 / np.pi)
    if os.environ.get('FUSED_GELU'):
        # WHAT-IF: the DeepProve scale-baked FUSED gelu (build_gelu_table_fused), IF it fired here.
        # ONE lookup = dequant(fc*s_in) -> gelu -> requant(/s_out) -> clamp[-L,L].  No rescale_to_P,
        # no separate sym_quant_act.  s_in is the c_fc calib scale (== the Reshape-input scale, since
        # Reshape is scale-transparent -- this is the "trace back through data-movement" fix).
        s_out = cs(calib_out)
        x = fc_int.astype(np.float64) * s_in
        gv = 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x ** 3)))
        return np.clip(rha(gv / s_out), -L, L).astype(np.int64), s_out
    # GENERIC (non-fused) gelu ref, the path the token run actually took: the folded graph has a
    # Reshape between c_fc/Gemm and the tanh-gelu pattern, so the fused Gelu node's input isn't in
    # the calib -> the DeepProve fused table is never built, and the gelu runs as
    #   rescale_to_P (input scale -> 2^-P) -> build_gelu_table lookup (gelu at 2^-P) -> requant to
    #   the Mul_3 calib scale.  (The fused table gives 443/512; this generic path gives 512/512.)
    xr = REQ(fc_int, s_in / (1.0 / SP))            # rescale_to_P: ints at 2^-P
    x = xr.astype(np.float64) / SP
    g = rha(0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x ** 3))) * SP)   # table @2^-P, llround
    return requant_static(g, 1.0 / SP, calib_out)  # sym_quant_act -> Mul_3 calib scale

def mul_scalar(x_int, sx, const, calib_name):
    # Mul ref: qm=llround(const*2^P); raw=x*qm; out=round_half_up(raw/2^P); scale unchanged; requant.
    qm = int(rha(np.array([const * SP]))[0])
    out = rqP(np.asarray(x_int, np.int64) * qm)
    return requant_static(out, sx, calib_name)

def softmax_lse(scores_int, s_scores, causal_mask):
    # rescale scores -> 2^-P (per-tensor single c,e), apply causal mask, then DeepProve LSE ref.
    c, e = scale_search(s_scores / (1.0 / SP))
    xr = rqu(scores_int, c, e).astype(np.int64)
    xr = np.where(causal_mask, xr, np.int64(-10 ** 12))   # masked -> huge neg -> exp 0
    S = float(SP)
    m = xr.max(-1, keepdims=True)
    diff = (m - xr).astype(np.float64)                    # >= 0
    summ = np.exp(-diff / S).sum(-1, keepdims=True)       # masked terms -> 0
    shift = np.maximum(rha(np.log(summ) * S), 0)          # round-half-away, clamp >= 0
    idx = diff + shift.astype(np.float64)
    out = rha(np.exp(-idx / S) * S)
    out = np.maximum(out, 0)                              # no-clamp table: masked/tail -> 0
    return out, 1.0 / SP

# ---------- forward ----------
def forward(R, ids, H=12):
    d = R['transformer.wte.weight'].shape[1]; dh = d // H; S = len(ids)
    pos = np.arange(S)
    # embeddings: Gather commits table @2^-P (llround), then static requant to the Gather calib
    wte_pre = rha(R['transformer.wte.weight'][ids] * SP)
    wte_q, s_wte = requant_static(wte_pre, 1.0 / SP, '/transformer/wte/Gather_output_0')
    wpe_pre = rha(R['transformer.wpe.weight'][pos] * SP)
    wpe_q, s_wpe = requant_static(wpe_pre, 1.0 / SP, '/transformer/wpe/Gather_output_0')
    # Add(wte, wpe) -> /transformer/Add_output_0
    x, sx = add_resid(wte_q, s_wte, wpe_q, s_wpe, '/transformer/Add_output_0')

    causal = np.tril(np.ones((S, S), bool))[None]          # [1,S,S] broadcast over heads
    for l in range(H):
        p = f'transformer.h.{l}.'; g = f'/transformer/h.{l}/'
        # ---- attention ----
        h, sh = layernorm(x, d, g + 'ln_1/LayerNormalization_output_0', R[p + 'ln_1.weight'], R[p + 'ln_1.bias'])
        qkv, sqkv = matmul(h, sh, R[p + 'attn.c_attn.weight'], R[p + 'attn.c_attn.bias'],
                           g + 'attn/c_attn/Gemm_output_0')
        q, k, v = [qkv[:, i * d:(i + 1) * d].reshape(S, H, dh).transpose(1, 0, 2) for i in range(3)]
        qs, sqm = mul_scalar(q, sqkv, 0.35355338, g + 'attn/Mul_output_0')     # Q * dh^-1/4
        ks, skm = mul_scalar(k, sqkv, 0.35355338, g + 'attn/Mul_1_output_0')   # K * dh^-1/4
        sc_acc = imm3(qs, ks.transpose(0, 2, 1))                              # QK^T [H,S,S]
        scores, s_sc = requant_static(sc_acc, sqm * skm, g + 'attn/MatMul_output_0')
        attn, s_at = softmax_lse(scores, s_sc, causal)                        # probs @2^-P
        av_acc = imm3(attn, v)                                                # attn @ V [H,S,dh]
        av, s_av = requant_static(av_acc, s_at * sqkv, g + 'attn/MatMul_1_output_0')
        a = av.transpose(1, 0, 2).reshape(S, d)                              # -> [S,d]
        ao, sao = matmul(a, s_av, R[p + 'attn.c_proj.weight'], R[p + 'attn.c_proj.bias'],
                         g + 'attn/c_proj/Gemm_output_0')
        x, sx = add_resid(ao, sao, x, sx, g + 'Add_output_0')                 # attn residual (ao first)
        # ---- mlp ----
        h, sh = layernorm(x, d, g + 'ln_2/LayerNormalization_output_0', R[p + 'ln_2.weight'], R[p + 'ln_2.bias'])
        fc, sfc = matmul(h, sh, R[p + 'mlp.c_fc.weight'], R[p + 'mlp.c_fc.bias'],
                         g + 'mlp/c_fc/Gemm_output_0')
        go, sg = gelu(fc, sfc, g + 'mlp/act/Mul_3_output_0')
        mo, smo = matmul(go, sg, R[p + 'mlp.c_proj.weight'], R[p + 'mlp.c_proj.bias'],
                         g + 'mlp/c_proj/Gemm_output_0')
        x, sx = add_resid(x, sx, mo, smo, g + 'Add_1_output_0')               # mlp residual (x first)

    h, sh = layernorm(x, d, '/transformer/ln_f/LayerNormalization_output_0',
                      R['transformer.ln_f.weight'], R['transformer.ln_f.bias'])
    acc, sacc = matmul(h, sh, R['lm_head_quarot'], None, None)               # logits_pre matmul
    Q, s_lp = requant_static(acc, sacc, 'logits_pre')                        # requant to logits_pre calib
    MCLAMP = 1 << (P + 8)                                                     # bias-add branch clamp
    bias = np.clip(rha(R['lm_bias_quarot'] / s_lp), -MCLAMP, MCLAMP)         # + per-vocab bias @ s_lp
    return Q + bias[None]

if __name__ == '__main__':
    R = load('model/gpt2-folded-s512.onnx')
    ids = [100 + i for i in range(512)]
    am = forward(R, ids).argmax(-1).tolist()
    print("mimic first 12:", am[:12])
    if os.environ.get("DUMP"): open(os.environ["DUMP"], "w").write(" ".join(map(str, am)))
    tok = sys.argv[1] if len(sys.argv) > 1 else '/tmp/zk_tokens.txt'
    if os.path.exists(tok):
        zk = [int(t) for t in open(tok).read().split()]
        n = min(len(zk), len(am))
        eq = [a == b for a, b in zip(am[:n], zk[:n])]
        agree = sum(eq)
        first = next((i for i, e in enumerate(eq) if not e), -1)
        print(f"vs ZK prover ({n} tokens): agree {agree}/{n}  ({100*agree/n:.1f}%)")
        if first >= 0:
            print(f"first mismatch at pos {first}: mimic={am[first]} zk={zk[first]}")
        else:
            print("BIT-EXACT: 512/512")
