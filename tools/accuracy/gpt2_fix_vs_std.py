# GPT-2 fix<->std column for Table 8: fixed-point proof (plaintext_fixed_gpt2, P sweep) vs the
# STANDARD int8 model (ORT dynamic quantization). Pools 10 x 512-token WikiText-2 windows.
# Also recomputes fix<->fp32 (must match existing Table 8) and std acc / std<->fp32 for the caption.
#   STD=<gpt2_int8_dyn.onnx> PYTHONPATH=. python3 tools/gpt2_fix_vs_std.py
import os, numpy as np, onnxruntime as ort
from transformers import GPT2Tokenizer
import tools.mimics.plaintext_fixed_gpt2 as MF
MODEL='model/gpt2-12_nocache/model.onnx'; STD=os.environ['STD']
W_LEN,K=512,10; Ps=[8,10,12,14]
tok=GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache')
allids=tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())
fp=ort.InferenceSession(MODEL,providers=['CPUExecutionProvider'])
st=ort.InferenceSession(STD,providers=['CPUExecutionProvider'])
Wf=MF.load(MODEL)
tot=0; cfp=cst=cst_fp=0
c_facc={P:0 for P in Ps}; c_ffp={P:0 for P in Ps}; c_fstd={P:0 for P in Ps}
for i in range(K):
    ids=allids[i*W_LEN:(i+1)*W_LEN]; S=len(ids)
    feed={'input_ids':np.array([ids]),'attention_mask':np.ones((1,S),np.int64),'position_ids':np.arange(S)[None]}
    fpp=fp.run(None,feed)[0][0].argmax(-1)[:-1]
    stp=st.run(None,feed)[0][0].argmax(-1)[:-1]
    gold=np.array(ids[1:]); n=len(gold); tot+=n
    cfp+=int((fpp==gold).sum()); cst+=int((stp==gold).sum()); cst_fp+=int((stp==fpp).sum())
    for P in Ps:
        MF.P=P; fx=MF.forward(Wf,ids,L=12,H=12).argmax(-1)[:-1]
        c_facc[P]+=int((fx==gold).sum()); c_ffp[P]+=int((fx==fpp).sum()); c_fstd[P]+=int((fx==stp).sum())
    print(f'[window {i+1}/{K}] done, cum={tot}', flush=True)
print(f'\n=== GPT-2 fix<->std over {tot} positions ===')
print(f'  fp32 acc {cfp/tot:.1%}   std-int8 acc {cst/tot:.1%}   std<->fp32 {cst_fp/tot:.1%}')
print(f'  {"P":>3}  {"fix acc":>7}  {"fix<->fp32":>10}  {"fix<->std":>9}')
for P in Ps:
    print(f'  {P:>3}  {c_facc[P]/tot:>6.1%}  {c_ffp[P]/tot:>9.1%}  {c_fstd[P]/tot:>8.1%}', flush=True)
