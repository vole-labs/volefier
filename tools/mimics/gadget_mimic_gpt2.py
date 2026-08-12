# Gadget-faithful GPT-2 mimic: replicates the ml-fixed ZK gadgets' EXACT integer/LUT
# arithmetic (sqrt-free dinv LayerNorm, exp-LUT Softmax, LUT Gelu) so we can swap each gadget
# in one at a time and see which one breaks the (fp32-matching) idealized result -> localizes
# the divergence between the ZK proof and fp32.
import sys, numpy as np, onnx
from onnx import numpy_helper as nh
P = 7
USE = {'ln': 'float', 'softmax': 'float', 'gelu': 'float'}   # per-gadget: 'float' (idealized) or 'gadget' (ZK-faithful)

def load(p): return {t.name: nh.to_array(t).astype(np.float64) for t in onnx.load(p).graph.initializer}
def s(): return float(1 << P)
def rdiv(a, D):                                   # round-half-away integer divide (== C rdiv / requant)
    a = np.asarray(a, dtype=np.int64); h = D // 2
    return np.where(a >= 0, (a + h) // D, -(((-a) + h) // D))
def toi(x): return np.where(x >= 0, np.floor(x * s() + 0.5), np.ceil(x * s() - 0.5)).astype(np.int64)   # real->int @2^-P
def Q(x): return toi(x).astype(np.float64) / s()

# ---- idealized (float) gadgets ----
def ln_f(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return Q(Q((x - mu) / np.sqrt(v + eps)) * Q(g) + Q(b))
def gelu_f(x): return Q(0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))))
def softmax_f(x):
    m = x.max(-1, keepdims=True); e = Q(np.exp(Q(x - m))); return Q(e / e.sum(-1, keepdims=True))

# ---- gadget-faithful (ZK) gadgets ----
def ln_g(x, g, b):                                # sqrt-free dinv LayerNorm (no eps), integer arithmetic
    xi = toi(x); d = xi.shape[-1]
    mu = rdiv(xi.sum(-1, keepdims=True), d); c = xi - mu
    vq = rdiv((c.astype(np.int64) ** 2).sum(-1, keepdims=True), d)         # var_q @ 2^-2P
    dinv = np.where(vq > 0, np.round((1 << (2 * P)) / np.sqrt(np.maximum(vq, 1))), 0).astype(np.int64)
    y = rdiv(c * dinv, 1 << P)                                            # normalized @ 2^-P
    gi, bi = toi(g), toi(b)
    return (rdiv(y * gi, 1 << P) + bi).astype(np.float64) / s()
_EXP = None; _GEL = None; _GOFF = None
def _tables():
    global _EXP, _GEL, _GOFF
    T = 1 << (P + 9); _EXP = np.maximum(np.round(np.exp(-np.arange(T) / s()) * s()), 1).astype(np.int64)
    GT = 1 << (P + 7); _GOFF = 1 << (P + 6); k = np.sqrt(2/np.pi)
    xx = (np.arange(GT) - _GOFF) / s(); _GEL = np.round(0.5 * xx * (1 + np.tanh(k * (xx + 0.044715 * xx**3))) * s()).astype(np.int64)
def gelu_g(x):
    xi = np.clip(toi(x) + _GOFF, 0, len(_GEL) - 1); return _GEL[xi].astype(np.float64) / s()
def softmax_g(x):
    xi = toi(x); mx = xi.max(-1, keepdims=True); diff = np.clip(mx - xi, 0, len(_EXP) - 1)
    e = _EXP[diff]; sm = e.sum(-1, keepdims=True)
    return rdiv(e * (1 << P), sm).astype(np.float64) / s()

def mm(a, w, b=None):
    y = Q(a) @ Q(w)
    if b is not None: y = y + Q(b)
    return Q(y)
def LN(x, g, bb): return (ln_g if USE['ln'] == 'gadget' else ln_f)(x, g, bb)
def GE(x): return (gelu_g if USE['gelu'] == 'gadget' else gelu_f)(x)
def SM(x): return (softmax_g if USE['softmax'] == 'gadget' else softmax_f)(x)

def forward(W, ids, L=6, H=12):
    d = W['transformer.wte.weight'].shape[1]; dh = d // H; S = len(ids)
    x = Q(W['transformer.wte.weight'][ids] + W['transformer.wpe.weight'][np.arange(S)])
    cm = np.tril(np.ones((S, S)))
    for l in range(L):
        p = f'transformer.h.{l}.'
        h = LN(x, W[p+'ln_1.weight'], W[p+'ln_1.bias'])
        qkv = mm(h, W[p+'attn.c_attn.weight'], W[p+'attn.c_attn.bias'])
        q, k, v = qkv[:, :d], qkv[:, d:2*d], qkv[:, 2*d:]
        q = q.reshape(S, H, dh).transpose(1,0,2); k = k.reshape(S, H, dh).transpose(1,0,2); v = v.reshape(S, H, dh).transpose(1,0,2)
        sc = Q(Q(q @ k.transpose(0,2,1)) / np.sqrt(dh)); sc = np.where(cm[None] > 0, sc, -1e9)
        a = (SM(sc) @ Q(v)); a = Q(a).transpose(1,0,2).reshape(S, d)
        x = Q(x + mm(a, W[p+'attn.c_proj.weight'], W[p+'attn.c_proj.bias']))
        h = LN(x, W[p+'ln_2.weight'], W[p+'ln_2.bias'])
        x = Q(x + mm(GE(mm(h, W[p+'mlp.c_fc.weight'], W[p+'mlp.c_fc.bias'])), W[p+'mlp.c_proj.weight'], W[p+'mlp.c_proj.bias']))
    x = LN(x, W['transformer.ln_f.weight'], W['transformer.ln_f.bias'])
    return mm(x, W['transformer.wte.weight'].T)

if __name__ == "__main__":
    W = load('model/gpt2-12_nocache/model.onnx'); ids = [464,3139,286,4881,318]; fp = [262,1748,262,338,783]
    _tables()
    print(f"fp32 target: {fp}")
    for cfg in [{}, {'ln':'gadget'}, {'softmax':'gadget'}, {'gelu':'gadget'}, {'ln':'gadget','softmax':'gadget','gelu':'gadget'}]:
        USE.update({'ln':'float','softmax':'float','gelu':'float'}); USE.update(cfg)
        am = forward(W, ids).argmax(-1).tolist()
        lbl = '+'.join(k for k,v in cfg.items()) or 'all-float (idealized)'
        print(f"  {lbl:32s}: {am}  {'== fp32' if am==fp else 'DIVERGES'}")
