# QuaRot-style rotation for GPT-2 (path B): rotate the residual stream by ONE orthonormal Q
# (folded into weights, free at proving) + an online Hadamard on the MLP intermediate (after gelu,
# before c_proj) and on the attention value/context. Goal: make int8 weights recover fp32 with a
# SHARED rotation (not per-matrix), at P>=10. Validated in plaintext before porting to the prover.
#
# Exactness relies on: (a) absorb LayerNorm affine (gamma,beta) into the next linear so LN->LN0
# (affine-free); (b) Q is orthonormal AND fixes the all-ones vector 1 (Q@1=1), so LN0 commutes:
# LN0(x@Q) = LN0(x)@Q; (c) the MLP/attn Hadamards H satisfy H@H^T=I and are applied online.
import numpy as np, onnx
from onnx import numpy_helper as nh
from scipy.linalg import hadamard

def load(p): return {t.name: nh.to_array(t).astype(np.float64) for t in onnx.load(p).graph.initializer}

def Q_fix_ones(d, seed=0):
    # orthonormal d x d with Q @ 1 = 1: identity on span(1), random rotation on its complement.
    rng = np.random.RandomState(seed)
    B, _ = np.linalg.qr(np.c_[np.ones(d), rng.randn(d, d-1)])   # B[:,0] ~ 1/sqrt(d), rest orthonormal complement
    R = np.eye(d); R[1:,1:], _ = np.linalg.qr(rng.randn(d-1, d-1))
    Q = B @ R @ B.T
    return Q

def Had(n):                                   # normalized Hadamard padded to next power of 2 (orthonormal on padded dim)
    N = 1
    while N < n: N *= 2
    return hadamard(N).astype(np.float64)/np.sqrt(N), N

P = 10
def s(): return float(1 << P)
def toi(x): return np.where(x >= 0, np.floor(x*s()+0.5), np.ceil(x*s()-0.5)).astype(np.int64)
def Qz(x): return toi(x).astype(np.float64)/s()
def rdiv(a, D): a=np.asarray(a,np.int64); h=D//2; return np.where(a>=0,(a+h)//D,-(((-a)+h)//D))

INT8=True; WMAX=127; MLP_HAD=True   # MLP_HAD: online Hadamard on gelu output (smooths c_proj 3072-input); off => pure weight fold
def mm(a, w, b=None, quant=True):
    if not (INT8 and quant):
        y=Qz(a)@Qz(w); return Qz(y+Qz(b)) if b is not None else Qz(y)
    sw=max(np.abs(w).max()/WMAX,1e-12); wq=np.clip(np.round(w/sw),-WMAX,WMAX)
    raw=toi(a).astype(np.int64)@wq.astype(np.int64)
    if b is not None: raw=raw+np.round(b/((2.0**-P)*sw)).astype(np.int64)
    return np.round(raw*sw).astype(np.float64)/s()
def ln0(x):                                   # affine-free LayerNorm (mean-sub + normalize)
    xi=toi(x); d=xi.shape[-1]; mu=rdiv(xi.sum(-1,keepdims=True),d); c=xi-mu
    vq=rdiv((c.astype(np.int64)**2).sum(-1,keepdims=True),d)
    dinv=np.where(vq>0,np.round((1<<(2*P))/np.sqrt(np.maximum(vq,1))),0).astype(np.int64)
    return rdiv(c*dinv,1<<P).astype(np.float64)/s()
_G=None;_GO=None
def tables():
    global _G,_GO
    GT=1<<(P+7); _GO=1<<(P+6); xx=(np.arange(GT)-_GO)/s()
    _G=np.round(0.5*xx*(1+np.tanh(np.sqrt(2/np.pi)*(xx+0.044715*xx**3)))*s()).astype(np.int64)
def gelu(x): return _G[np.clip(toi(x)+_GO,0,len(_G)-1)].astype(np.float64)/s()
def softmax(x):
    xi=toi(x); mx=xi.max(-1,keepdims=True); diff=np.clip(mx-xi,0,(1<<(P+9))-1)
    e=np.maximum(np.round(np.exp(-diff/s())*s()),1).astype(np.int64)
    return rdiv(e*(1<<P),e.sum(-1,keepdims=True)).astype(np.float64)/s()

def rotate_weights(W, L, H=12, rot=True):
    """Fold the QuaRot residual rotation into a copy of the weights. ALWAYS absorbs LayerNorm
    affine (so the forward uses affine-free ln0); rot=False uses Q=I (pure absorption, exact baseline)."""
    d=W['transformer.wte.weight'].shape[1]
    R={k:v.copy() for k,v in W.items()}
    Q=Q_fix_ones(d,0) if rot else np.eye(d)
    Hm,Nm=Had(4*d)
    R['transformer.wte.weight']=W['transformer.wte.weight']@Q
    R['transformer.wpe.weight']=W['transformer.wpe.weight']@Q
    for l in range(L):
        p=f'transformer.h.{l}.'
        for ln,lin in [('ln_1','attn.c_attn'),('ln_2','mlp.c_fc')]:
            g=W[p+ln+'.weight']; b=W[p+ln+'.bias']; Wt=W[p+lin+'.weight']; bt=W[p+lin+'.bias']
            R[p+lin+'.bias']=bt + b@Wt                              # absorb beta (uses ORIGINAL W)
            R[p+lin+'.weight']=Q.T@(g[:,None]*Wt)                   # absorb gamma into rows, rotate input by Q^T
            R[p+ln+'.weight']=np.ones_like(g); R[p+ln+'.bias']=np.zeros_like(b)   # LN -> LN0
        R[p+'attn.c_proj.weight']=W[p+'attn.c_proj.weight']@Q       # attn output -> residual: rotate weight AND bias by Q
        R[p+'attn.c_proj.bias']=W[p+'attn.c_proj.bias']@Q
        Wp=W[p+'mlp.c_proj.weight']; n=Wp.shape[0]
        if rot:                                                    # c_proj: input(4d) gets H_mlp^T (online), output gets Q
            Wpp=np.zeros((Nm,Wp.shape[1])); Wpp[:n]=Wp; R[p+'mlp.c_proj.weight']=(Hm.T@Wpp)@Q
        else:
            R[p+'mlp.c_proj.weight']=Wp@Q
        R[p+'mlp.c_proj.bias']=W[p+'mlp.c_proj.bias']@Q
    gf=W['transformer.ln_f.weight']; bf=W['transformer.ln_f.bias']
    R['__lmhead__']=Q.T@(gf[:,None]*W['transformer.wte.weight'].T)  # final LN_f absorbed into LM head [d, vocab]
    R['__lmbias__']=bf@W['transformer.wte.weight'].T
    return R, (Hm,Nm), Q

def forward(W, ids, L, rot=True, H=12, lm_quant=False):
    R, HM, Q = rotate_weights(W, L, H, rot)
    d=R['transformer.wte.weight'].shape[1]; dh=d//H; S=len(ids)
    x=Qz(R['transformer.wte.weight'][ids]+R['transformer.wpe.weight'][np.arange(S)])
    cm=np.tril(np.ones((S,S)))
    for l in range(L):
        p=f'transformer.h.{l}.'
        h=ln0(x)
        qkv=mm(h,R[p+'attn.c_attn.weight'],R[p+'attn.c_attn.bias'])
        q,k,v=qkv[:,:d],qkv[:,d:2*d],qkv[:,2*d:]
        q=q.reshape(S,H,dh).transpose(1,0,2);k=k.reshape(S,H,dh).transpose(1,0,2);v=v.reshape(S,H,dh).transpose(1,0,2)
        sc=Qz(Qz(q@k.transpose(0,2,1))/np.sqrt(dh)); sc=np.where(cm[None]>0,sc,-1e9)
        a=(softmax(sc)@Qz(v)); a=Qz(a).transpose(1,0,2).reshape(S,d)
        x=Qz(x+mm(a,R[p+'attn.c_proj.weight'],R[p+'attn.c_proj.bias']))
        h=ln0(x)
        m=gelu(mm(h,R[p+'mlp.c_fc.weight'],R[p+'mlp.c_fc.bias']))
        if rot:                                                    # online H on gelu out (pad to N), then c_proj
            Hm,Nm=HM; mp=np.zeros((S,Nm)); mp[:,:m.shape[1]]=m; m=Qz(mp@Hm)
        x=Qz(x+mm(m,R[p+'mlp.c_proj.weight'],R[p+'mlp.c_proj.bias']))
    h=ln0(x)
    return mm(h,R['__lmhead__'],R['__lmbias__'],quant=lm_quant)

if __name__=="__main__":
    ids=[464,3139,286,4881,318]; tgt=[198,286,262,11,262]
    W=load('model/gpt2-12_nocache/model.onnx')
    P=10; tables()
    INT8=False; fr=forward(W,ids,12,rot=True).argmax(-1).tolist()
    print(f"12-layer GPT-2, fp32 target {tgt}   (P={P})")
    print(f"  fold exactness (fp + QuaRot)     : {fr}  {'EXACT' if fr==tgt else 'BUG'}")
    INT8=True
    for wb,bits in [(127,8),(255,9),(511,10)]:
        WMAX=wb
        nr=forward(W,ids,12,rot=False).argmax(-1).tolist(); rr=forward(W,ids,12,rot=True).argmax(-1).tolist()
        print(f"  int{bits}: no-rot {str(nr):26s} {'=fp32' if nr==tgt else '     '}   + QuaRot {str(rr):26s} {'=fp32' if rr==tgt else ''}")
