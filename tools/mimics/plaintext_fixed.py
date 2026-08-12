# Plaintext fixed-point (2^-P) forward mirroring ml-fixed's FixedGraphInfer EXACTLY:
# qz(v)=round(v*2^P); conv/gemm = rdiv(sum(x_q*w_q), 2^P)+qz(bias);
# BN = rdiv(x_q*qz(scale/sqrt(var+eps)), 2^P)+qz(beta-g*mean); relu=max(0,x); add=x+y.
import sys, os, numpy as np, onnx
from onnx import numpy_helper
P=int(os.environ.get("P","12")); SC=1<<P    # fixed-point precision bits; override with env P
def qz(v):                                  # round half AWAY from zero (== C++ llround), not banker's
    v=np.asarray(v,np.float64)*SC
    return np.where(v>=0,np.floor(v+0.5),np.ceil(v-0.5)).astype(np.int64)
def rdiv(a,D=SC):
    a=np.asarray(a,np.int64); h=D//2
    return np.where(a>=0,(a+h)//D,-(((-a)+h)//D))
def run(model,x_f32):
    g=onnx.load(model).graph; init={t.name:numpy_helper.to_array(t) for t in g.initializer}
    act={}; inp=[i.name for i in g.input if i.name not in init][0]
    act[inp]=qz(x_f32.reshape(1,3,224,224))
    A=lambda n: act[n]
    for nd in g.node:
        op=nd.op_type; I=nd.input; O=nd.output[0]
        at={a.name:(list(a.ints) if a.ints else (a.f if a.type==1 else a.i)) for a in nd.attribute}
        if op=="Conv":
            X=A(I[0]).reshape(A(I[0]).shape[-3:]); W=qz(init[I[1]]); b=qz(init[I[2]]) if len(I)>2 else None
            Cout,Cin,kh,kw=W.shape; st=at.get("strides",[1,1]); pd=at.get("pads",[0,0,0,0])
            C,H,Wd=X.shape; Xp=np.pad(X,((0,0),(pd[0],pd[2]),(pd[1],pd[3])))
            Ho=(H+pd[0]+pd[2]-kh)//st[0]+1; Wo=(Wd+pd[1]+pd[3]-kw)//st[1]+1
            o=np.zeros((Cout,Ho,Wo),np.int64)
            for r in range(kh):
                for c in range(kw):
                    sub=Xp[:,r:r+Ho*st[0]:st[0],c:c+Wo*st[1]:st[1]][:, :Ho,:Wo]
                    o+=np.tensordot(W[:,:,r,c],sub,axes=([1],[0]))
            o=rdiv(o);
            if b is not None: o+=b[:,None,None]
            act[O]=o[None]
        elif op=="BatchNormalization":
            X=A(I[0]); sc,be,mn,vr=[init[I[k]].astype(np.float64) for k in (1,2,3,4)]; eps=at.get("epsilon",1e-5)
            gg=sc/np.sqrt(vr+eps); qg=qz(gg); qb=qz(be-gg*mn)
            sh=[1,len(qg),1,1] if X.ndim==4 else [1,len(qg)]
            act[O]=rdiv(X*qg.reshape(sh))+qb.reshape(sh)
        elif op=="Relu": act[O]=np.maximum(0,A(I[0]))
        elif op=="Add":
            if I[0] in init or I[1] in init:            # FC bias: act + constant initializer (qz at 2^P), matches FixedGraphInfer
                ain=I[1] if I[0] in init else I[0]; cst=I[0] if I[0] in init else I[1]
                a=A(ain); bq=qz(init[cst]).ravel(); sh=[1,-1] if a.ndim==2 else [1,len(bq),1,1]
                act[O]=a+bq.reshape(sh)
            else: act[O]=A(I[0])+A(I[1])
        elif op=="MaxPool":
            X=A(I[0]).reshape(A(I[0]).shape[-3:]); C,H,Wd=X.shape
            kh,kw=at.get("kernel_shape",[2,2]); st=at.get("strides",[2,2]); pd=at.get("pads",[0,0,0,0])
            Xp=np.pad(X,((0,0),(pd[0],pd[2]),(pd[1],pd[3])),constant_values=-(1<<62))
            Ho=(H+pd[0]+pd[2]-kh)//st[0]+1; Wo=(Wd+pd[1]+pd[3]-kw)//st[1]+1; o=np.full((C,Ho,Wo),-(1<<62),np.int64)
            for r in range(kh):
                for c in range(kw):
                    o=np.maximum(o,Xp[:,r:r+Ho*st[0]:st[0],c:c+Wo*st[1]:st[1]][:, :Ho,:Wo])
            act[O]=o[None]
        elif op in("GlobalAveragePool","AveragePool"):
            X=A(I[0]).reshape(A(I[0]).shape[-3:]); C,H,Wd=X.shape
            if op=="GlobalAveragePool": act[O]=rdiv(X.reshape(C,-1).sum(1),H*Wd)[None,:,None,None]
            else:
                kh,kw=at.get("kernel_shape",[2,2]); st=at.get("strides",[2,2]); Ho=(H-kh)//st[0]+1; Wo=(Wd-kw)//st[1]+1
                o=np.zeros((C,Ho,Wo),np.int64)
                for r in range(kh):
                    for c in range(kw): o+=X[:,r:r+Ho*st[0]:st[0],c:c+Wo*st[1]:st[1]][:, :Ho,:Wo]
                act[O]=rdiv(o,kh*kw)[None]
        elif op in("Gemm","MatMul"):
            X=A(I[0]).reshape(1,-1); W=qz(init[I[1]]); b=qz(init[I[2]]) if len(I)>2 and I[2] in init else 0
            tb=at.get("transB",0); Wm=W if tb else W.T   # rows=out
            act[O]=rdiv(X@Wm.T)+b
        elif op in("Flatten","Reshape","Squeeze","Unsqueeze","Dropout","Identity"): act[O]=A(I[0])
        else: raise RuntimeError("unhandled "+op)
    return act[g.output[0].name].ravel().astype(np.int64)
if __name__=="__main__":
    o=run(sys.argv[1],np.fromfile(sys.argv[2],np.float32)); print("plaintext fixed ARGMAX =",int(o.argmax()))
    if len(sys.argv)>3:
        ref=np.fromfile(sys.argv[3],np.int64); n=min(len(o),len(ref)); d=np.abs(o[:n]-ref[:n]).max()
        print(f"vs ZK out_ref: max|diff|={d} argmax(zk)={int(ref.argmax())} {'BIT-EXACT MATCH' if d==0 else 'diff'}")
