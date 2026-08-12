# Plaintext fixed-point GPT-2 forward (numpy), mirroring the ml-fixed scheme:
# every value lives at scale 2^-P; after each op the result is re-quantized to 2^-P with
# round-half-away (== C llround / the rdiv gadget). P=None => exact fp32 (matches ORT).
# Used to (a) check the fixed-point argmax vs fp32, and (b) sweep P to find where they agree.
import sys, numpy as np, onnx
from onnx import numpy_helper as nh

P = None  # set per-run
def Q(x):
    if P is None: return x
    s = 2.0 ** P
    return np.where(x >= 0, np.floor(x * s + 0.5), np.ceil(x * s - 0.5)) / s   # round-half-away to 2^-P

def load(path):
    m = onnx.load(path); return {t.name: nh.to_array(t).astype(np.float64) for t in m.graph.initializer}

def ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(v + eps)
    return Q(Q(y) * Q(g) + Q(b))

def gelu(x):                                    # gelu_new (tanh approx)
    return Q(0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))))

def softmax(x):
    m = x.max(-1, keepdims=True); e = Q(np.exp(Q(x - m)))
    return Q(e / e.sum(-1, keepdims=True))

def mm(a, w, b=None):                            # fixed-point matmul: requant result to 2^-P
    y = Q(a) @ Q(w)
    if b is not None: y = y + Q(b)
    return Q(y)

def forward(W, ids, L=6, H=12):
    d = W['transformer.wte.weight'].shape[1]; dh = d // H; S = len(ids)
    x = Q(W['transformer.wte.weight'][ids] + W['transformer.wpe.weight'][np.arange(S)])
    cmask = np.tril(np.ones((S, S))); neg = -1e9
    for l in range(L):
        p = f'transformer.h.{l}.'
        h = ln(x, W[p + 'ln_1.weight'], W[p + 'ln_1.bias'])
        qkv = mm(h, W[p + 'attn.c_attn.weight'], W[p + 'attn.c_attn.bias'])       # [S,3d]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        q = q.reshape(S, H, dh).transpose(1, 0, 2); k = k.reshape(S, H, dh).transpose(1, 0, 2); v = v.reshape(S, H, dh).transpose(1, 0, 2)
        sc = Q(Q(q @ k.transpose(0, 2, 1)) / np.sqrt(dh))                         # [H,S,S]
        sc = np.where(cmask[None] > 0, sc, neg)
        a = softmax(sc) @ Q(v)                                                    # [H,S,dh]
        a = Q(a).transpose(1, 0, 2).reshape(S, d)
        x = Q(x + mm(a, W[p + 'attn.c_proj.weight'], W[p + 'attn.c_proj.bias']))
        h = ln(x, W[p + 'ln_2.weight'], W[p + 'ln_2.bias'])
        h = gelu(mm(h, W[p + 'mlp.c_fc.weight'], W[p + 'mlp.c_fc.bias']))
        x = Q(x + mm(h, W[p + 'mlp.c_proj.weight'], W[p + 'mlp.c_proj.bias']))
    x = ln(x, W['transformer.ln_f.weight'], W['transformer.ln_f.bias'])
    logits = mm(x, W['transformer.wte.weight'].T)                                 # tied LM head
    return logits

if __name__ == "__main__":
    W = load('model/gpt2-12_nocache/model.onnx'); ids = [100, 101, 102, 103]
    P = None; fp = forward(W, ids).argmax(-1).tolist()
    print(f"fp32 (P=None) argmax: {fp}   <- ground truth")
    print(f"{'P':>4} | argmax                | matches fp32?")
    for p in [7, 8, 10, 12, 14, 16, 20]:
        P = p; am = forward(W, ids).argmax(-1).tolist()
        print(f"{p:>4} | {str(am):20s} | {'YES' if am == fp else 'no'}")
