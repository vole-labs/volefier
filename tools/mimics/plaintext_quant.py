# Plaintext int8 forward that mirrors ml-quant's QuantGraphInfer EXACTLY
# (same scale_search c*2^e + round-half-up requant, same zero-point-folded conv).
# Validated bit-exact against the ZK out_ref; runs fast for accuracy sweeps.
import sys, numpy as np, onnx
from onnx import numpy_helper

def scale_search(M):                       # mirror QuantInfer::scale_search
    best=1e300; bc,be=1,0
    for e in range(-24,11):
        for c in range(1,801):
            d=abs(np.ldexp(float(c),e)-M)
            if d<best: best,bc,be=d,c,e
    return bc,be
_cache={}
FLOAT_SCALE=False     # False = ml-quant (ZK fixed-point c*2^e requant); True = ideal int8 (exact float scale)
def requant(a, M):
    a=a.astype(np.int64)
    if FLOAT_SCALE:                         # round_half_away(a * M) -- exact float requant multiplier
        v=a.astype(np.float64)*M
        return np.where(v>=0,np.floor(v+0.5),np.ceil(v-0.5)).astype(np.int64)
    if M not in _cache: _cache[M]=scale_search(M)   # round_half_UP(c*a / 2^shift) -- ZK c*2^e
    c,e=_cache[M]; num=c*a
    if e>=0: return num<<e                          # upscale ratio>=1: exact integer multiply (requant_cm_one guard)
    sh=-e; D=1<<sh
    return (num+D//2)//D   # round-half-UP (floor of biased); matches QuantInfer::requant_cm_one (was round-half-away)

def run(model_path, x_uint8, in_scale, in_zp):
    m=onnx.load(model_path); g=m.graph
    init={t.name:numpy_helper.to_array(t) for t in g.initializer}
    act={}; scale={}; zp={}; cref={}; cs={}; czp={}
    inp=[i.name for i in g.input if i.name not in init][0]
    # input uint8 is at (data_scale, data_zp) -- mirrors QuantGraphInfer::run seeding
    act[inp]=x_uint8.astype(np.int64); scale[inp]=in_scale; zp[inp]=in_zp
    def A(n): return act[n]
    for nd in g.node:
        op=nd.op_type; I=nd.input; O=nd.output[0]
        if op=="DequantizeLinear":
            s=float(init[I[1]]); z=int(init[I[2]]) if len(I)>2 else 0
            if I[0] in init:        # weight/bias constant -> keep int values
                cref[O]=init[I[0]].astype(np.int64); cs[O]=s; czp[O]=z
            else:
                act[O]=A(I[0]); scale[O]=s; zp[O]=z
        elif op=="QuantizeLinear":
            s=float(init[I[1]]); z=int(init[I[2]]) if len(I)>2 else 0
            v=A(I[0]); M=scale[I[0]]/s
            q=requant((v-zp[I[0]]).astype(np.int64),M)+z
            act[O]=np.clip(q,0,255); scale[O]=s; zp[O]=z
        elif op=="Conv":
            X=A(I[0]); W=cref[I[1]]; b=cref[I[2]].astype(np.int64) if len(I)>2 else None
            sx=scale[I[0]]; xz=zp[I[0]]; Cout,Cin,kh,kw=W.shape
            strides=[1,1]; pads=[0,0,0,0]
            for at in nd.attribute:
                if at.name=="strides": strides=list(at.ints)
                if at.name=="pads": pads=list(at.ints)
            X=X.reshape(X.shape[-3:]) if X.ndim==4 else X  # CHW
            C,H,Wd=X.shape; Xp=np.pad(X,((0,0),(pads[0],pads[2]),(pads[1],pads[3])))
            Ho=(H+pads[0]+pads[2]-kh)//strides[0]+1; Wo=(Wd+pads[1]+pads[3]-kw)//strides[1]+1
            out=np.zeros((Cout,Ho,Wo),np.int64)
            for r in range(kh):
                for c in range(kw):
                    sub=Xp[:,r:r+Ho*strides[0]:strides[0],c:c+Wo*strides[1]:strides[1]][:, :Ho,:Wo]  # C,Ho,Wo
                    out+=np.tensordot(W[:,:,r,c],sub,axes=([1],[0]))  # Cout,Ho,Wo
            ws=W.reshape(Cout,-1).sum(1)                      # zero-point fold
            beff=(b if b is not None else 0)-xz*ws
            out+=beff[:,None,None]
            act[O]=out[None]; scale[O]=sx*cs[I[1]]; zp[O]=0
        elif op in ("MaxPool","GlobalAveragePool"):
            X=A(I[0]); X=X.reshape(X.shape[-3:]); C,H,Wd=X.shape
            if op=="GlobalAveragePool":
                s=X.reshape(C,-1).sum(1); cnt=H*Wd
                act[O]=(np.where(s>=0,(s+cnt//2)//cnt,-(((-s)+cnt//2)//cnt)))[None,:,None,None]
            else:
                kh=kw=2; st=2; pads=[0,0,0,0]
                for at in nd.attribute:
                    if at.name=="kernel_shape": kh,kw=at.ints
                    if at.name=="strides": st=at.ints[0]
                    if at.name=="pads": pads=list(at.ints)
                Xp=np.pad(X,((0,0),(pads[0],pads[2]),(pads[1],pads[3])),constant_values=-(1<<62))  # -inf: padded cells excluded from max
                Ho=(H+pads[0]+pads[2]-kh)//st+1; Wo=(Wd+pads[1]+pads[3]-kw)//st+1; o=np.full((C,Ho,Wo),-1<<62,np.int64)
                for r in range(kh):
                    for c in range(kw):
                        o=np.maximum(o,Xp[:,r:r+Ho*st:st,c:c+Wo*st:st][:, :Ho,:Wo])
                act[O]=o[None]
            scale[O]=scale[I[0]]; zp[O]=zp[I[0]]
        elif op=="Add":
            if I[0] in init or I[1] in init:                       # FC bias: act + constant initializer
                ain=I[1] if I[0] in init else I[0]; cst=I[0] if I[0] in init else I[1]
                a=A(ain); sa=scale[ain]; T=init[cst]
                if T.dtype in (np.float32,np.float64):
                    t=T.astype(np.float64).ravel()/sa; bq=np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5)).astype(np.int64)
                else: bq=T.astype(np.int64).ravel()
                sh=[1,-1] if a.ndim==2 else [1,len(bq),1,1]
                act[O]=a+bq.reshape(sh); scale[O]=sa; zp[O]=zp[ain]
            else:
                a=A(I[0]); b=A(I[1]); sa,sb=scale[I[0]],scale[I[1]]   # residual: (a-za)+requant(b-zb), zp=0
                ai=a-zp[I[0]]; bi=requant((b-zp[I[1]]).astype(np.int64),sb/sa)
                act[O]=ai+bi; scale[O]=sa; zp[O]=0
        elif op in ("Gemm","MatMul"):
            X=A(I[0]).reshape(1,-1); W=cref[I[1]]; b=cref[I[2]].astype(np.int64) if len(I)>2 else 0
            tb=any(at.name=="transB" and at.i for at in nd.attribute)
            Wm=W if (op=="Gemm" and tb) else W.T
            xz=zp[I[0]]; out=(X-xz)@Wm.T.astype(np.int64)+b
            act[O]=out; scale[O]=scale[I[0]]*cs[I[1]]; zp[O]=0
        elif op=="Relu":                                          # max(zp, x_q): dequant>=0 <=> x_q>=zp
            act[O]=np.maximum(zp[I[0]],A(I[0])); scale[O]=scale[I[0]]; zp[O]=zp[I[0]]
        elif op in ("Flatten","Reshape","Squeeze","Unsqueeze","Dropout","Identity"):
            act[O]=A(I[0]); scale[O]=scale[I[0]]; zp[O]=zp[I[0]]
        else: raise RuntimeError("unhandled op "+op)
    out=act[g.output[0].name].ravel().astype(np.int64)
    return out

if __name__=="__main__":
    model=sys.argv[1]; xf=sys.argv[2]
    x=np.fromfile(xf,np.float32)
    m=onnx.load(model); init={t.name:numpy_helper.to_array(t) for t in m.graph.initializer}
    if "data_scale" in init:                    # same params the C++ test reads
        ds=float(init["data_scale"]); dz=int(init["data_zero_point"])
    else:
        ds=dz=None
        for nd in m.graph.node:
            if nd.op_type in ("QuantizeLinear","DequantizeLinear") and nd.input[0]=="data":
                ds=float(init[nd.input[1]]); dz=int(init[nd.input[2]]); break
    t=x/ds                                   # round half AWAY from zero (== C++ lround), not banker's
    xq=np.clip(np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5))+dz,0,255).astype(np.int64).reshape(1,3,224,224)
    o=run(model,xq,ds,dz); am=int(o.argmax())
    print("plaintext int8 ARGMAX =",am)
    if len(sys.argv)>3:
        ref=np.fromfile(sys.argv[3],np.int64)
        n=min(len(o),len(ref)); diff=np.abs(o[:n]-ref[:n]).max()
        print(f"vs ZK oref: max|diff|={diff}  argmax(zk)={int(ref.argmax())}  {'BIT-EXACT MATCH' if diff==0 else 'MISMATCH'}")
