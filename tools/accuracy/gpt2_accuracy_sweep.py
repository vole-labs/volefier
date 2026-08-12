import sys, os, numpy as np, onnxruntime as ort
from transformers import GPT2Tokenizer
import tools.mimics.plaintext_fixed_gpt2 as MF        # ml-fixed mimic (uniform 2^-P, round-half-away) -- ZK-faithful
import tools.mimics.prover_exact_mimic as MQ          # ml-quant mimic (QuaRot int12, round-half-up c*2^e) -- prover-exact
from tools.rotate_fold.quarot_gpt2 import load as load_q
MODEL='model/gpt2-12_nocache/model.onnx'
N=int(sys.argv[1]) if len(sys.argv)>1 else 128
P=int(sys.argv[2]) if len(sys.argv)>2 else 12

tok=GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache')
ids=tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())[:N]
S=len(ids)

# fp32 baseline (ORT)
sess=ort.InferenceSession(MODEL,providers=['CPUExecutionProvider'])
fp=sess.run(None,{'input_ids':np.array([ids]),'attention_mask':np.ones((1,S),np.int64),
                  'position_ids':np.arange(S)[None]})[0][0]
fp_am=fp.argmax(-1)

# ml-fixed (Test 1): plain gpt2-12 at uniform P
Wf=MF.load(MODEL); MF.P=P
fix_am=MF.forward(Wf,ids,L=12,H=12).argmax(-1)

# ml-quant (Test 2): QuaRot int12 prover-exact
Wq=load_q(MODEL)
qa_am=MQ.forward(Wq,ids,H=12).argmax(-1)

# next-token: position i predicts ids[i+1]
gold=np.array(ids[1:]); fpp,fixp,qap=fp_am[:-1],fix_am[:-1],qa_am[:-1]; n=len(gold)
print(f"\n=== GPT-2 (12L) / WikiText-2, next-token over {n} positions  (ml-fixed P={P}) ===")
print(f"  fp32 (ORT, BASELINE)   acc-vs-gold = {np.mean(fpp==gold):.1%}")
print(f"  ml-fixed (P={P})         acc-vs-gold = {np.mean(fixp==gold):.1%}   agree-with-fp32 = {np.mean(fixp==fpp):.1%}")
print(f"  ml-quant (int12 QuaRot) acc-vs-gold = {np.mean(qap==gold):.1%}   agree-with-fp32 = {np.mean(qap==fpp):.1%}")
diff=np.where(fixp!=qap)[0]
print(f"\n  ml-fixed vs ml-quant DIFFER on {len(diff)} / {n} next-token predictions (DIRECT count)")
