# GPT-2 analog of the CNN Table-6 sweep, with the PROVER-EXACT deployed model.
# Fixed-point proof (uniform 2^-P, plaintext_fixed_gpt2) vs the deployed QuaRot int-B model
# (prover_exact_mimic, bit-exact to the ZK prover), on WikiText-2 next-token.
# Reports fp32/quant accuracy, quant<->fp32, fix<->fp32, and fix<->quant(int-B) (completeness).
import sys, numpy as np, onnxruntime as ort
from transformers import GPT2Tokenizer
import tools.mimics.plaintext_fixed_gpt2 as MF
import tools.mimics.prover_exact_mimic as MQ          # deployed: prover-exact QuaRot int-B (set_bits(B))
MODEL='model/gpt2-12_nocache/model.onnx'
N=int(sys.argv[1]) if len(sys.argv)>1 else 256
Ps=[8,10,12,14]; BITS=[8,10,12]
tok=GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache')
ids=tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())[:N]; S=len(ids)
sess=ort.InferenceSession(MODEL,providers=['CPUExecutionProvider'])
fp=sess.run(None,{'input_ids':np.array([ids]),'attention_mask':np.ones((1,S),np.int64),
                  'position_ids':np.arange(S)[None]})[0][0].argmax(-1)
gold=np.array(ids[1:]); fpp=fp[:-1]; n=len(gold)
Wq=MQ.load(MODEL); quant={}
for b in BITS:
    MQ.set_bits(b); quant[b]=MQ.forward(Wq,ids,H=12).argmax(-1)[:-1]
Wf=MF.load(MODEL)
print(f"\n=== GPT-2 (12L) / WikiText-2, next-token over {n} positions  (deployed = prover-exact QuaRot int-B) ===")
print(f"  fp32 accuracy = {np.mean(fpp==gold):.1%}")
for b in BITS:
    print(f"  deployed int{b}: accuracy = {np.mean(quant[b]==gold):.1%}   quant<->fp32 = {np.mean(quant[b]==fpp):.1%}")
print(f"  {'P':>3}  {'acc':>7}  {'fix<->fp32':>10}  " + "  ".join(f'fix<->q{b:<2}' for b in BITS))
for P in Ps:
    MF.P=P; fx=MF.forward(Wf,ids,L=12,H=12).argmax(-1)[:-1]
    print(f"  {P:>3}  {np.mean(fx==gold):>6.1%}  {np.mean(fx==fpp):>9.1%}  " + "  ".join(f'{np.mean(fx==quant[b]):>7.1%}' for b in BITS), flush=True)
