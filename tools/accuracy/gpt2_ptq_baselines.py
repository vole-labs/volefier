# GPT-2 quantization BASELINE comparison on WikiText-2 (argmax agreement vs fp32).
# CORRECTED: a PROPER standard PTQ uses per-channel weights + per-token activations + a
# high-precision LM head (the embedding matrix has huge dynamic range; per-tensor int8 there
# destroys the logits). The naive per-tensor/int8-LM-head version collapses (~15%) and is a
# strawman -- done properly, standard integer PTQ reaches ~95% agree-with-fp32.
#   STANDARD PTQ  : per-channel int8 W, per-token int8 A, fp32 nonlinears + fp32 LM head
#   WEIGHT-ONLY   : int8 weights, fp32 activations + nonlinears
#   OURS          : int-N weights + 2^-P fixed-point activations + gadget LUT nonlinears (the ZK prover)
#   PYTHONPATH=. python3 tools/gpt2_ptq_baselines.py
import numpy as np, onnxruntime as ort, tools.mimics.plaintext_int8_gpt2 as M
from transformers import GPT2Tokenizer
tok=GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache'); W=M.load('model/gpt2-12_nocache/model.onnx'); g=lambda k:W['transformer.'+k]
s=ort.InferenceSession('model/gpt2-12_nocache/model.onnx',providers=['CPUExecutionProvider'])
def ln(x,ga,be,eps=1e-5): mu=x.mean(-1,keepdims=True); v=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(v+eps)*ga+be
def gelu(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def smax(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def mmq(a,w,b=None):                          # per-token int8 act, per-channel int8 weight, exact fp32 dequant
    sa=np.maximum(np.abs(a).max(-1,keepdims=True)/127,1e-12); aq=np.clip(np.round(a/sa),-127,127)
    sw=np.maximum(np.abs(w).max(0,keepdims=True)/127,1e-12);   wq=np.clip(np.round(w/sw),-127,127)
    y=(aq@wq)*sa*sw; return y+b if b is not None else y
def mmw(a,w,b=None):                          # weight-only int8 (per-channel), fp32 activations
    sw=np.maximum(np.abs(w).max(0,keepdims=True)/127,1e-12); y=a@(np.clip(np.round(w/sw),-127,127)*sw); return y+b if b is not None else y
# --- SYMMETRIC int12 (the ZK prover's sym_act_bits=12 scheme): per-tensor POWER-OF-2 int12 activations
#     (the sym_quant_act hook), per-tensor int12 weights, softmax + LM-head left full-precision. zp=0. ---
L=2047
def q_pow2(a):                                # CURRENT prover: per-tensor POWER-OF-2 int12
    m=float(np.abs(a).max());
    return a if m<1e-30 else np.clip(np.round(a/(2.0**np.ceil(np.log2(m/L)))),-2048,2047)*(2.0**np.ceil(np.log2(m/L)))
def q_arb(a):                                 # per-tensor ARBITRARY-scale int12 (s=max/L)
    s=max(float(np.abs(a).max())/L,1e-12); return np.clip(np.round(a/s),-L,L)*s
def q_tok(a):                                 # per-TOKEN (per-row) arbitrary int12 -> handles per-position outliers
    s=np.maximum(np.abs(a).max(-1,keepdims=True)/L,1e-12); return np.clip(np.round(a/s),-L,L)*s
def mm12(a,w,b=None,pc=False):                # int12 weight: per-tensor or PER-CHANNEL (pc)
    sw=(np.maximum(np.abs(w).max(0,keepdims=True)/L,1e-12) if pc else max(float(np.abs(w).max())/L,1e-12))
    y=a@(np.clip(np.round(w/sw),-L,L)*sw); return y+b if b is not None else y
def block12(ids,q,pc=False,lm_fp=False):
    H=12;d=768;dh=64;S=len(ids); x=q(g('wte.weight')[ids]+g('wpe.weight')[np.arange(S)]); cm=np.tril(np.ones((S,S)))
    for l in range(12):
        p=f'h.{l}.'; h=q(ln(x,g(p+'ln_1.weight'),g(p+'ln_1.bias')))
        qkv=q(mm12(h,g(p+'attn.c_attn.weight'),g(p+'attn.c_attn.bias'),pc)); q_,k,v=[t.reshape(S,H,dh).transpose(1,0,2) for t in (qkv[:,:d],qkv[:,d:2*d],qkv[:,2*d:])]
        sc=(q_@k.transpose(0,2,1))/np.sqrt(dh); sc=np.where(cm[None]>0,sc,-1e9); a=q((smax(sc)@v).transpose(1,0,2).reshape(S,d))  # softmax NOT quantized
        x=q(x+mm12(a,g(p+'attn.c_proj.weight'),g(p+'attn.c_proj.bias'),pc)); h2=q(ln(x,g(p+'ln_2.weight'),g(p+'ln_2.bias')))
        x=q(x+mm12(q(gelu(mm12(h2,g(p+'mlp.c_fc.weight'),g(p+'mlp.c_fc.bias'),pc))),g(p+'mlp.c_proj.weight'),g(p+'mlp.c_proj.bias'),pc))
    hf=q(ln(x,g('ln_f.weight'),g('ln_f.bias')));
    return (hf@g('wte.weight').T) if lm_fp else mm12(hf,g('wte.weight').T,pc=True)  # LM head: per-channel int12 (huge wte dynamic range) or fp
def block(ids,mm,lm_fp32):
    H=12;d=768;dh=64;S=len(ids); x=g('wte.weight')[ids]+g('wpe.weight')[np.arange(S)]; cm=np.tril(np.ones((S,S)))
    for l in range(12):
        p=f'h.{l}.'; h=ln(x,g(p+'ln_1.weight'),g(p+'ln_1.bias'))
        qkv=mm(h,g(p+'attn.c_attn.weight'),g(p+'attn.c_attn.bias')); q,k,v=[t.reshape(S,H,dh).transpose(1,0,2) for t in (qkv[:,:d],qkv[:,d:2*d],qkv[:,2*d:])]
        sc=(q@k.transpose(0,2,1))/np.sqrt(dh); sc=np.where(cm[None]>0,sc,-1e9); a=(smax(sc)@v).transpose(1,0,2).reshape(S,d)
        x=x+mm(a,g(p+'attn.c_proj.weight'),g(p+'attn.c_proj.bias')); h2=ln(x,g(p+'ln_2.weight'),g(p+'ln_2.bias'))
        x=x+mm(gelu(mm(h2,g(p+'mlp.c_fc.weight'),g(p+'mlp.c_fc.bias'))),g(p+'mlp.c_proj.weight'),g(p+'mlp.c_proj.bias'))
    hf=ln(x,g('ln_f.weight'),g('ln_f.bias')); return (hf@g('wte.weight').T) if lm_fp32 else mm(hf,g('wte.weight').T)

from scipy.special import log_softmax
w=tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())[:400]
fwl=s.run(None,{'input_ids':np.array([w]),'attention_mask':np.ones((1,len(w)),np.int64),'position_ids':np.arange(len(w))[None]})[0][0]
fw=fwl.argmax(-1); nx=np.array(w[1:])
def metrics(lg):                              # agree-fp32 | top-1 vs actual | perplexity | cosine-sim vs fp32 logits (DeepProve's metric)
    am=lg.argmax(-1); ppl=float(np.exp((-log_softmax(lg,-1)[np.arange(len(w)-1),nx]).mean()))
    cos=float(np.mean(np.sum(lg*fwl,1)/(np.linalg.norm(lg,axis=1)*np.linalg.norm(fwl,axis=1)+1e-12)))
    return 100*(am==fw).mean(), 100*(am[:-1]==nx).mean(), ppl, 100*cos
print(f'WikiText-2 ({len(w)} positions)              agree-fp32 | top-1 | perplexity | cosine-fp32')
ag,t1,pp,co=metrics(fwl); print(f'  fp32 (reference)                          :    100%   |  {t1:.0f}%  |   {pp:5.1f}   |  {co:.2f}%')
for name,fn in [('STANDARD PTQ (per-ch W8,per-tok A8,fp32 LMhd)', lambda: block(w,mmq,True)),
                ('WEIGHT-ONLY int8 (fp32 activations)',            lambda: block(w,mmw,True))]:
    ag,t1,pp,co=metrics(fn()); print(f'  {name:42s}:    {ag:.0f}%   |  {t1:.0f}%  |   {pp:5.1f}   |  {co:.2f}%')
for P,wb in [(7,10),(12,10)]:
    M.P=P; M.WBITS=wb; M.tables(); ag,t1,pp,co=metrics(M.forward(W,w,12,last_only=False))
    print(f'  OURS P={P:<2d} int{wb} (2^-P act + gadget nonlin)     :    {ag:.0f}%   |  {t1:.0f}%  |   {pp:5.1f}   |  {co:.2f}%')
for name,q,pc,lm in [('int12 per-tensor 2^k act, per-tensor W (CURRENT prover)', q_pow2, False, False),
                     ('int12 per-tensor ARBITRARY-scale act, per-tensor W   ', q_arb,  False, False),
                     ('int12 per-tensor ARB act, PER-CHANNEL W              ', q_arb,  True,  False),
                     ('int12 PER-TOKEN act, PER-CHANNEL W                   ', q_tok,  True,  False),
                     ('int12 PER-TOKEN act, PER-CHANNEL W, fp32 LM head     ', q_tok,  True,  True)]:
    ag,t1,pp,co=metrics(block12(w,q,pc,lm)); print(f'  OURS {name}:  {ag:3.0f}%  | {t1:3.0f}% | {pp:6.1f} | {co:6.2f}%')
print('  [DeepProve reports >=99.6% cosine-fp32 at 12-bit GPT-2 (per-tensor calibrated + outlier smoothing)]')
