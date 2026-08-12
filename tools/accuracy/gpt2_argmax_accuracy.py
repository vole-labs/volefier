# GPT-2 argmax-accuracy harness: measures how the ml-quant per-tensor int-N scheme
# (the plaintext mimic = what the ZK prover computes) compares to fp32 (ORT), over a
# real dataset instead of one prompt.
#   LAMBADA  -> next-word ACCURACY (argmax at last context pos == gold first target token)
#   WikiText -> per-token AGREEMENT (argmax matches fp32 at every position)
# Reports, per bit-width: absolute accuracy AND agreement-with-fp32 (the ZK-relevant metric).
#
#   python3 -m tools.gpt2_argmax_accuracy [n_lambada] [n_wiki_tokens]
import sys, json, numpy as np, onnxruntime as ort
import tools.mimics.plaintext_int8_gpt2 as M
from transformers import GPT2Tokenizer

MODEL='model/gpt2-12_nocache/model.onnx'
N_LAMBADA = int(sys.argv[1]) if len(sys.argv)>1 else 300
N_WIKI    = int(sys.argv[2]) if len(sys.argv)>2 else 512
CONFIGS = [(7,10),(10,10),(12,10),(7,8)]   # (P activation-precision, WBITS weight-bits); prover uses P=7, int10

tok = GPT2Tokenizer.from_pretrained('model/gpt2-12_nocache')
W = M.load(MODEL)
sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
def cfg(P,wb): M.P=P; M.WBITS=wb; M.tables()

def fp32_logits(ids):                       # ORT, full fp32; returns logits [S, vocab]
    S=len(ids)
    return sess.run(None,{'input_ids':np.array([ids]),'attention_mask':np.ones((1,S),np.int64),
                          'position_ids':np.arange(S)[None]})[0][0]

# ---------------- LAMBADA: next-word accuracy ----------------
ex=[json.loads(l)['text'] for l in open('model/eval/gpt2_eval/lambada_test.jsonl')][:N_LAMBADA]
cases=[]
for text in ex:
    full=tok.encode(text); tgt=tok.encode(' '+text.rstrip().rsplit(' ',1)[-1])
    if len(tgt)>=len(full): continue
    ctx=full[:-len(tgt)];
    if len(ctx)>=2: cases.append((ctx, full[len(ctx)]))     # (context, gold first target token)
fp_am=[int(fp32_logits(c)[-1].argmax()) for c,_ in cases]
gold=[g for _,g in cases]
print(f"=== LAMBADA next-word accuracy ({len(cases)} examples) ===")
print(f"  fp32                 : {100*np.mean([a==g for a,g in zip(fp_am,gold)]):.1f}% correct")
for P,wb in CONFIGS:
    cfg(P,wb); am=[int(M.forward(W,c,12,last_only=True)[-1].argmax()) for c,_ in cases]
    acc=100*np.mean([a==g for a,g in zip(am,gold)]); agr=100*np.mean([a==f for a,f in zip(am,fp_am)])
    print(f"  P={P:<2d} int{wb:<2d} (prover) : {acc:.1f}% correct  |  {agr:.1f}% agree-with-fp32")

# ---------------- WikiText-2: per-token argmax agreement ----------------
wids=tok.encode(open('model/eval/gpt2_eval/wikitext2_test.txt').read())[:N_WIKI]
fp=fp32_logits(wids).argmax(-1); nxt=np.array(wids[1:])
print(f"\n=== WikiText-2 per-token agreement ({len(wids)} positions) ===")
print(f"  fp32 top-1 vs actual : {100*(fp[:-1]==nxt).mean():.1f}%")
for P,wb in CONFIGS:
    cfg(P,wb); q=M.forward(W,wids,12,last_only=False).argmax(-1)
    print(f"  P={P:<2d} int{wb:<2d}        : {100*(q==fp).mean():.1f}% agree-with-fp32  |  {100*(q[:-1]==nxt).mean():.1f}% top-1 vs actual")
