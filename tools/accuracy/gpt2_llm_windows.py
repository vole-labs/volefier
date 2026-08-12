# Windowed GPT-2 completeness test: tile WikiText-2 into K non-overlapping W_LEN-token windows,
# pool per-token counts across windows. Deployed = prover-exact QuaRot int-B; fixed = uniform 2^-P.
#   python3 tools/gpt2_llm_windows.py [W_LEN=512] [K=10]
import sys, numpy as np, onnxruntime as ort
from transformers import GPT2Tokenizer
import tools.mimics.plaintext_fixed_gpt2 as MF
import tools.mimics.prover_exact_mimic as MQ
MODEL='model/gpt2-12_nocache/model.onnx'
W_LEN=int(sys.argv[1]) if len(sys.argv)>1 else 512
K=int(sys.argv[2]) if len(sys.argv)>2 else 10
Ps=[8,10,12,14]; BITS=[8,10,12]
tok=GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache')
allids=tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())
windows=[allids[i*W_LEN:(i+1)*W_LEN] for i in range(K)]
sess=ort.InferenceSession(MODEL,providers=['CPUExecutionProvider'])
Wf=MF.load(MODEL); Wq=MQ.load(MODEL)
tot=0; c_fp=0
c_qacc={b:0 for b in BITS}; c_qfp={b:0 for b in BITS}
c_facc={P:0 for P in Ps}; c_ffp={P:0 for P in Ps}
c_fq={(P,b):0 for P in Ps for b in BITS}
for wi,ids in enumerate(windows):
    S=len(ids)
    fp=sess.run(None,{'input_ids':np.array([ids]),'attention_mask':np.ones((1,S),np.int64),
                      'position_ids':np.arange(S)[None]})[0][0].argmax(-1)
    gold=np.array(ids[1:]); fpp=fp[:-1]; n=len(gold); tot+=n; c_fp+=int((fpp==gold).sum())
    quant={}
    for b in BITS:
        MQ.set_bits(b); qb=MQ.forward(Wq,ids,H=12).argmax(-1)[:-1]; quant[b]=qb
        c_qacc[b]+=int((qb==gold).sum()); c_qfp[b]+=int((qb==fpp).sum())
    for P in Ps:
        MF.P=P; fx=MF.forward(Wf,ids,L=12,H=12).argmax(-1)[:-1]
        c_facc[P]+=int((fx==gold).sum()); c_ffp[P]+=int((fx==fpp).sum())
        for b in BITS: c_fq[(P,b)]+=int((fx==quant[b]).sum())
    print(f"[window {wi+1}/{K}] done, cum positions={tot}", flush=True)
print(f"\n=== GPT-2 pooled over {K} x {W_LEN} = {tot} next-token positions ===")
print(f"  fp32 accuracy = {c_fp/tot:.1%}")
for b in BITS: print(f"  deployed int{b}: accuracy = {c_qacc[b]/tot:.1%}   quant<->fp32 = {c_qfp[b]/tot:.1%}")
print(f"  {'P':>3}  {'acc':>7}  {'fix<->fp32':>10}  " + "  ".join(f'fix<->q{b:<2}' for b in BITS))
for P in Ps:
    print(f"  {P:>3}  {c_facc[P]/tot:>6.1%}  {c_ffp[P]/tot:>9.1%}  " + "  ".join(f'{c_fq[(P,b)]/tot:>7.1%}' for b in BITS), flush=True)
