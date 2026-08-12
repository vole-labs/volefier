# Plaintext PER-TENSOR INT8 GPT-2 forward, mirroring the ml-quant scheme (zkGPT/DeepProve):
#   weights -> symmetric int8 (s_w = max|W|/127); activations at 2^-P (c=1);
#   matmul: raw = sum a_q*w_q (+ bias/(s_x*s_w)); requant out = round(raw * s_w) @ 2^-P.
# Nonlinears use the ZK-faithful gadgets (sqrt-free dinv LN, exp-LUT softmax, gelu-LUT) so this
# tracks the ZK ml-quant computation. Used to localize the 12-layer mac-fail: does the int8
# *scheme* diverge from fp32 (-> scheme bug) or match it (-> pure ZK-commitment bug)?
import sys, numpy as np, onnx
from onnx import numpy_helper as nh
P = 7
def load(p): return {t.name: nh.to_array(t).astype(np.float64) for t in onnx.load(p).graph.initializer}
def s(): return float(1 << P)
def toi(x): return np.where(x >= 0, np.floor(x*s()+0.5), np.ceil(x*s()-0.5)).astype(np.int64)
def Q(x): return toi(x).astype(np.float64)/s()
def rdiv(a, D): a=np.asarray(a,np.int64); h=D//2; return np.where(a>=0,(a+h)//D,-(((-a)+h)//D))

# ---- ZK-faithful nonlinears (same as gadget_mimic) ----
def ln_g(x,g,b):
    xi=toi(x); d=xi.shape[-1]; mu=rdiv(xi.sum(-1,keepdims=True),d); c=xi-mu
    vq=rdiv((c.astype(np.int64)**2).sum(-1,keepdims=True),d)
    dinv=np.where(vq>0,np.round((1<<(2*P))/np.sqrt(np.maximum(vq,1))),0).astype(np.int64)
    y=rdiv(c*dinv,1<<P); return (rdiv(y*toi(g),1<<P)+toi(b)).astype(np.float64)/s()
_E=None;_G=None;_GO=None
def tables():
    global _E,_G,_GO
    T=1<<(P+9); _E=np.maximum(np.round(np.exp(-np.arange(T)/s())*s()),1).astype(np.int64)
    GT=1<<(P+8); _GO=1<<(P+7); xx=(np.arange(GT)-_GO)/s()      # gelu domain +-128 (matches ml-quant impl: gtab 1<<15, off 1<<14)
    _G=np.round(0.5*xx*(1+np.tanh(np.sqrt(2/np.pi)*(xx+0.044715*xx**3)))*s()).astype(np.int64)
def gelu_g(x): return _G[np.clip(toi(x)+_GO,0,len(_G)-1)].astype(np.float64)/s()
def softmax_g(x):
    xi=toi(x); mx=xi.max(-1,keepdims=True); diff=np.clip(mx-xi,0,len(_E)-1); e=_E[diff]
    return rdiv(e*(1<<P),e.sum(-1,keepdims=True)).astype(np.float64)/s()

# ---- PER-TENSOR INT-N matmul (the ml-quant scheme), optional outlier smoothing ----
WBITS=10         # weight bit-width: symmetric range +-(2^(WBITS-1)-1). MATCHES impl default weight_max=511 (int10); int8 too lossy >6 layers.
ROTATE=False     # DeepProve/QuaRot outlier smoothing: rotate each weight by an orthonormal M (y=(x@M)(M^T@W),
                 # exact since M M^T = I). Spreads outlier channels -> per-tensor int8 grid fits the bulk ->
                 # int8 recovers fp32 even at 12 layers. At proving time M folds into the previous layer (free).
def _Morth(n, seed):
    rng=np.random.RandomState(1000+seed); Qm,_=np.linalg.qr(rng.randn(n,n)); return Qm
def mm(a, w, b=None):
    WM=2**(WBITS-1)-1
    if ROTATE:
        M=_Morth(w.shape[0], w.shape[0]*7+w.shape[1]); a=a@M; w=M.T@w        # exact reparameterization
    sw=max(np.abs(w).max()/WM,1e-12); wq=np.clip(np.round(w/sw),-WM,WM)
    aq=toi(a); sx=2.0**-P
    raw=aq.astype(np.int64)@wq.astype(np.int64)
    if b is not None: raw=raw+np.round(b/(sx*sw)).astype(np.int64)
    return np.round(raw*sw).astype(np.float64)/s()   # requant to 2^-P (exact s_w; ZK uses c*2^e approx)

def forward(W, ids, L, H=12, last_only=False, lm_uniform=True):
    # last_only: return only the final-position logits (fast for next-token tasks).
    # lm_uniform: LM head at uniform 2^-P (matches the prover's matmul_wte), not int-N.
    d=W['transformer.wte.weight'].shape[1]; dh=d//H; S=len(ids)
    x=Q(W['transformer.wte.weight'][ids]+W['transformer.wpe.weight'][np.arange(S)])
    cm=np.tril(np.ones((S,S)))
    for l in range(L):
        p=f'transformer.h.{l}.'
        h=ln_g(x,W[p+'ln_1.weight'],W[p+'ln_1.bias'])
        qkv=mm(h,W[p+'attn.c_attn.weight'],W[p+'attn.c_attn.bias'])
        q,k,v=qkv[:,:d],qkv[:,d:2*d],qkv[:,2*d:]
        q=q.reshape(S,H,dh).transpose(1,0,2);k=k.reshape(S,H,dh).transpose(1,0,2);v=v.reshape(S,H,dh).transpose(1,0,2)
        sc=Q(Q(q@k.transpose(0,2,1))/np.sqrt(dh)); sc=np.where(cm[None]>0,sc,-1e9)
        a=(softmax_g(sc)@Q(v)); a=Q(a).transpose(1,0,2).reshape(S,d)
        x=Q(x+mm(a,W[p+'attn.c_proj.weight'],W[p+'attn.c_proj.bias']))
        h=ln_g(x,W[p+'ln_2.weight'],W[p+'ln_2.bias'])
        x=Q(x+mm(gelu_g(mm(h,W[p+'mlp.c_fc.weight'],W[p+'mlp.c_fc.bias'])),W[p+'mlp.c_proj.weight'],W[p+'mlp.c_proj.bias']))
    x=ln_g(x,W['transformer.ln_f.weight'],W['transformer.ln_f.bias'])
    xt = x[-1:] if last_only else x
    wteT = W['transformer.wte.weight'].T
    if lm_uniform: return Q(Q(xt) @ Q(wteT))      # uniform 2^-P LM head (prover's matmul_wte convention)
    return mm(xt, wteT)

if __name__=="__main__":
    ids=[464,3139,286,4881,318]; tables()
    W=load('model/gpt2-12_nocache/model.onnx'); fp=[198,286,262,11,262]
    print(f"12-layer GPT-2, fp32 target: {fp}")
    for wb,rot in [(8,False),(8,True),(10,False)]:
        WBITS=wb; ROTATE=rot                       # module globals read by mm()
        am=forward(W,ids,12).argmax(-1).tolist()
        tag=f"int{wb}{' +rotation(outlier-smoothing)' if rot else ''}"
        print(f"  {tag:34s}: {am}  {'== fp32' if am==fp else 'WRONG'}")
