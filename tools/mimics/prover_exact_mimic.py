# PROVER-EXACT int-tracking mimic of the QuaRot static-int12 GPT-2 prover.  Every activation is
# (int array, scale); every op replays the prover's EXACT gadget: scale_search -> c*2^e + round-HALF-UP
# requant, gelu LUT (P+8/P+7), integer LayerNorm (var+1, dinv=round(2^2P/sqrt(var))), exp-LUT softmax,
# integer attention + online Hadamard, per-tensor int12.  Goal: match the prover bit-for-bit, so its
# WikiText accuracy is the prover's TRUE accuracy (not the higher-precision baseline mimic's).
#   PYTHONPATH=. python3 tools/prover_exact_mimic.py            # 5-tok + 16-tok sanity (vs prover 5/5, 13/16)
#   PYTHONPATH=. python3 tools/prover_exact_mimic.py wiki       # WikiText accuracy
import sys, os, numpy as np
from tools.rotate_fold.quarot_gpt2 import rotate_weights, load
EXACT = bool(os.environ.get('EXACT'))    # exact float requant (test: is scale_search c<=800 the bottleneck?)
SN = bool(os.environ.get('SN'))          # scale-native softmax (per-tensor int12 scores, DeepProve-style)
CMAX = int(os.environ.get('CMAX', '800'))  # scale_search multiplier range (DeepProve fpm has >=12 bits)
NO_H = bool(os.environ.get('NO_HADAMARD'))  # Q-only (no online MLP Hadamard) -- mirrors export_quarot NO_HADAMARD=1

def set_bits(B):                               # B-bit static quant: P-bit bus, symmetric clamp +-(2^(B-1)-1)
    global P, L, SP, GT, GOFF, ET, _xx, GEL, EXP
    P = B; L = (1 << (P - 1)) - 1; SP = 1 << P
    GT = 1 << (P + 8); GOFF = 1 << (P + 7); ET = 1 << (P + 9)
    _xx = (np.arange(GT) - GOFF) / SP
    GEL = np.round(0.5 * _xx * (1 + np.tanh(np.sqrt(2/np.pi) * (_xx + 0.044715 * _xx**3))) * SP).astype(np.int64)
    EXP = np.maximum(np.round(np.exp(-np.arange(ET) / SP) * SP), 1).astype(np.int64)
set_bits(int(os.environ.get('BITS', '12')))    # default int12 (unchanged); BITS=8/10 -> int8/int10

def scale_search(target):                      # prover quant_infer.h: e in [-24,10], c in [1,800], min|c*2^e - target|
    best = 1e300; bc = 1; be = 0
    for e in range(-24, 11):
        c0 = int(round(target / (2.0**e)))
        for c in range(max(1, c0 - 1), min(CMAX, c0 + 1) + 1):
            d = abs(c * (2.0**e) - target)
            if d < best: best = d; bc = c; be = e
    return bc, be
def REQ(acc, ratio):                           # requant by an arbitrary ratio: scale_search c*2^e (prover) or EXACT float
    acc = np.asarray(acc, np.int64)
    if EXACT: return np.floor(acc.astype(np.float64) * ratio + 0.5).astype(np.int64)   # round-half-up, exact
    c, e = scale_search(ratio); return rqu(acc, c, e)

def rqu(acc, c, e):                            # requant_cm_one: round-half-up(acc*c*2^e)
    acc = np.asarray(acc, dtype=np.int64)
    if e >= 0: return acc * c * (1 << e)
    M = -e; N = acc * c
    return (N + (np.int64(1) << (M - 1))) >> M    # numpy int64 >> = arithmetic floor -> round-half-up
def rqP(N, M):                                 # round-half-up by M (c=1) -- LN normalize / shifts
    N = np.asarray(N, np.int64); return (N + (np.int64(1) << (M - 1))) >> M
def rdiv(a, D):                                # round-half-away (Div gadget / graph LN ref)
    a = np.asarray(a, np.int64); h = D // 2
    return np.where(a >= 0, (a + h) // D, -(((-a) + h) // D))

def qpt(X, s):                                 # quantize (X int @ s) -> per-tensor int12 (dynamic max = self-calib)
    so = max(float(np.abs(X).max()) * s / L, 1e-30); return REQ(X, s / so), so
def wq(Wfp):                                   # per-tensor symmetric int weight
    sw = max(float(np.abs(Wfp).max()) / L, 1e-30); return np.clip(np.round(Wfp / sw), -L, L).astype(np.int64), sw
def mm(X, sx, Wfp, bfp=None, to_pt=True):      # int matmul + (optional) per-tensor requant
    q, sw = wq(Wfp); acc = X.astype(np.int64) @ q; sacc = sx * sw
    if bfp is not None: acc = acc + np.round(bfp / sacc).astype(np.int64)
    return qpt(acc, sacc) if to_pt else (acc, sacc)
def gelu(X, sx):
    xp = REQ(X, sx * SP)                                            # -> 2^-P
    g = GEL[np.clip(xp + GOFF, 0, GT - 1)]                          # LUT
    return qpt(g, 1.0 / SP)
def lnorm(X, sx, d):                           # affine-free integer LayerNorm
    mu = rdiv(X.sum(-1, keepdims=True), d); c = X - mu
    vq = rdiv((c.astype(np.int64)**2).sum(-1, keepdims=True), d) + 1
    dinv = np.round((1 << (2*P)) / np.sqrt(vq)).astype(np.int64)
    return qpt(rqP(c * dinv, P), 1.0 / SP)
def add(A, sa, B, sb):                          # residual: align B to A's scale, add, re-quantize
    return qpt(A + REQ(B, sb / sa), sa)

def forward(W, ids, H=12):
    R, HM, Q = rotate_weights(W, 12, H, rot=True); Hm, Nm = HM
    Hsub = Hm[:R['transformer.h.0.mlp.c_proj.weight'].shape[0] - 0][:R['transformer.h.0.mlp.c_fc.weight'].shape[1]] if False else Hm
    dff = W['transformer.h.0.mlp.c_proj.weight'].shape[0]; Hs = Hm[:dff, :]   # online Hadamard [dff, Nm]
    Hq = np.sign(Hs).astype(np.int64) * L; sH = (1.0 / np.sqrt(Nm)) / L        # int Hadamard (+-L) and its scale
    d = R['transformer.wte.weight'].shape[1]; dh = d // H; S = len(ids)
    x, sx = qpt(np.round((R['transformer.wte.weight'][ids] + R['transformer.wpe.weight'][np.arange(S)]) * SP).astype(np.int64), 1.0/SP)
    cm = np.tril(np.ones((S, S)))
    for l in range(12):
        p = f'transformer.h.{l}.'
        h, sh = lnorm(x, sx, d)
        qkv, sqk = mm(h, sh, R[p+'attn.c_attn.weight'], R[p+'attn.c_attn.bias'])
        q, k, v = [qkv[:, i*d:(i+1)*d].reshape(S, H, dh).transpose(1, 0, 2) for i in range(3)]
        sc_acc = q.astype(np.int64) @ k.transpose(0, 2, 1)                      # int Q.K^T at sqk^2
        if SN:                                                                  # scale-native: per-tensor int12 scores + scale-baked exp LUT
            ssc = sqk * sqk / np.sqrt(dh)
            smax = max(float(np.abs(sc_acc).max()) * ssc / L, 1e-30)
            sci = np.where(cm[None] > 0, np.clip(np.round(sc_acc * ssc / smax), -L, L).astype(np.int64), -L)
            diff = np.clip(sci.max(-1, keepdims=True) - sci, 0, ET - 1)
            exb = np.maximum(np.round(np.exp(-np.arange(ET) * smax) * SP), 1).astype(np.int64)
            ex = exb[diff]
        else:
            sc = np.where(cm[None] > 0, REQ(sc_acc, sqk*sqk*SP/np.sqrt(dh)), -10**9)  # -> 2^-P, *1/sqrt(dh)
            diff = np.clip(sc.max(-1, keepdims=True) - sc, 0, ET - 1); ex = EXP[diff]
        attn = rdiv(ex * SP, ex.sum(-1, keepdims=True))                         # softmax at 2^-P
        a_acc = (attn.astype(np.int64) @ v).transpose(1, 0, 2).reshape(S, d)
        a, sa = qpt(a_acc, (1.0/SP) * sqk)
        ao, sao = mm(a, sa, R[p+'attn.c_proj.weight'], R[p+'attn.c_proj.bias'])
        x, sx = add(x, sx, ao, sao)
        h, sh = lnorm(x, sx, d)
        fc, sfc = mm(h, sh, R[p+'mlp.c_fc.weight'], R[p+'mlp.c_fc.bias'])
        g, sg = gelu(fc, sfc)
        if NO_H:                                                                # Q-only: c_proj.weight = W@Q (no H^T fold, no online Hadamard)
            mo, smo = mm(g, sg, W[p+'mlp.c_proj.weight'] @ Q, R[p+'mlp.c_proj.bias'])
        else:
            had, shad = qpt(g.astype(np.int64) @ Hq, sg * sH)                   # online Hadamard (public int)
            mo, smo = mm(had, shad, R[p+'mlp.c_proj.weight'], R[p+'mlp.c_proj.bias'])
        x, sx = add(x, sx, mo, smo)
    h, sh = lnorm(x, sx, d)
    lg, _ = mm(h, sh, R['__lmhead__'], R['__lmbias__'])
    return lg

if __name__ == '__main__':
    W = load('model/gpt2-12_nocache/model.onnx')
    if len(sys.argv) > 1 and sys.argv[1] == 'wiki':
        import onnxruntime as ort
        from transformers import GPT2Tokenizer
        from scipy.special import log_softmax
        tok = GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache')
        w = tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())[:int(os.environ.get('TOKENS','400'))]
        s = ort.InferenceSession('model/gpt2-12_nocache/model.onnx', providers=['CPUExecutionProvider'])
        fwl = s.run(None, {'input_ids': np.array([w]), 'attention_mask': np.ones((1, len(w)), np.int64), 'position_ids': np.arange(len(w))[None]})[0][0]
        fw = fwl.argmax(-1); nx = np.array(w[1:])
        lg = forward(W, w); am = lg.argmax(-1)
        ppl = float(np.exp((-log_softmax(lg, -1)[np.arange(len(w)-1), nx]).mean()))
        cos = float(np.mean(np.sum(lg*fwl, 1) / (np.linalg.norm(lg, axis=1)*np.linalg.norm(fwl, axis=1)+1e-12)))
        print(f'PROVER-EXACT mimic on WikiText-2 ({len(w)} pos): agree-fp32 {100*(am==fw).mean():.0f}%  top1 {100*(am[:-1]==nx).mean():.0f}%  ppl {ppl:.1f}  cos {100*cos:.2f}%')
    else:
        for ids, fp, tag in [([464,3139,286,4881,318], [198,286,262,11,262], 'prover 5/5'),
                             ([220+i*97 for i in range(16)], [198,347,64,898,11,32,677,1704,6,660,260,705,705,705,705,262], 'prover 13/16')]:
            pr = forward(W, ids).argmax(-1).tolist()
            print(f'  ids[{len(ids)}]: mimic {pr}')
            print(f'           fp32 {fp}  -> agree {sum(a==b for a,b in zip(pr,fp))}/{len(fp)}  ({tag})')
