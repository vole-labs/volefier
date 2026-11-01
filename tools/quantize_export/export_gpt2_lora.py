# Export HF GPT-2 small + a WikiText-2 token window for the LoRA fine-tuning proof
# (test/ml-fixed/lora_train.h, env GPT2_BIN / TOKENS).
#   [WIKITEXT_TXT=<train text>] python3 tools/quantize_export/export_gpt2_lora.py [out_dir] [seq] [offset]
# writes out_dir/gpt2_lora.bin and out_dir/tokens.txt (seq+1 GPT-2 token ids of the WikiText-2
# train split starting at token `offset`; position t is trained to predict token t+1).  For a
# mini-batch of B sequences of length L pass seq = B*L.
# gpt2_lora.bin: "GPT2LORA", int32 NB D H FFN V NPOS, then float32
#   wte[V,D] wpe[NPOS,D], per block ln1_g ln1_b Wq bq Wk bk Wv bv Wo bo ln2_g ln2_b W1 b1 W2 b2,
#   then lnf_g lnf_b.  Every matrix is stored dout x din (HF Conv1D keeps din x dout).
# Also prints the float model's mean next-token cross-entropy on the window and the activation
# ranges the fixed-point tables must cover.
import os, sys, struct
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

out = sys.argv[1] if len(sys.argv) > 1 else "model/gpt2_lora"
seq = int(sys.argv[2]) if len(sys.argv) > 2 else 512
offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(out, exist_ok=True)


def wikitext_train():
    if os.environ.get("WIKITEXT_TXT"):  # pre-fetched plain text of the train split
        return open(os.environ["WIKITEXT_TXT"]).read()
    try:
        from datasets import load_dataset
        return "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    except Exception as e:  # no `datasets`: read the parquet straight from the hub
        print(f"datasets unavailable ({e}); fetching the parquet from the hub")
        import pandas as pd
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                            repo_type="dataset")
        return "".join(pd.read_parquet(p)["text"])


tok = GPT2TokenizerFast.from_pretrained("gpt2")
ids = tok(wikitext_train())["input_ids"][offset:offset + seq + 1]
assert len(ids) == seq + 1
with open(os.path.join(out, "tokens.txt"), "w") as f:
    f.write(" ".join(map(str, ids)) + "\n")

model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
cfg, sd = model.config, model.state_dict()
D, NB, H, V, NPOS = cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.vocab_size, cfg.n_positions
FFN = sd["transformer.h.0.mlp.c_fc.weight"].shape[1]
a = lambda k: sd[k].detach().float().numpy()
with open(os.path.join(out, "gpt2_lora.bin"), "wb") as f:
    f.write(b"GPT2LORA" + struct.pack("<6i", NB, D, H, FFN, V, NPOS))
    w = lambda x: f.write(np.ascontiguousarray(x, dtype="<f4").tobytes())
    w(a("transformer.wte.weight")); w(a("transformer.wpe.weight"))
    for b in range(NB):
        p = f"transformer.h.{b}."
        ca, cb = a(p + "attn.c_attn.weight"), a(p + "attn.c_attn.bias")
        w(a(p + "ln_1.weight")); w(a(p + "ln_1.bias"))
        for k in range(3):  # Q, K, V column blocks of the fused c_attn
            w(ca[:, k * D:(k + 1) * D].T); w(cb[k * D:(k + 1) * D])
        w(a(p + "attn.c_proj.weight").T); w(a(p + "attn.c_proj.bias"))
        w(a(p + "ln_2.weight")); w(a(p + "ln_2.bias"))
        w(a(p + "mlp.c_fc.weight").T); w(a(p + "mlp.c_fc.bias"))
        w(a(p + "mlp.c_proj.weight").T); w(a(p + "mlp.c_proj.bias"))
    w(a("transformer.ln_f.weight")); w(a("transformer.ln_f.bias"))
print(f"wrote {out}/gpt2_lora.bin (NB={NB} D={D} H={H} FFN={FFN} V={V}) and {out}/tokens.txt ({seq + 1} ids)")

# float reference on the window (its first NPOS positions: GPT-2's context)
rng = {"pre_gelu": 0.0, "resid": 0.0}
hooks = [m.mlp.c_fc.register_forward_hook(lambda m, i, o: rng.__setitem__("pre_gelu", max(rng["pre_gelu"], o.abs().max().item())))
         for m in model.transformer.h]
n = min(seq, NPOS)
x = torch.tensor([ids[:n]])
with torch.no_grad():
    o = model(x, output_hidden_states=True)
    ce_t = torch.nn.functional.cross_entropy(o.logits[0], torch.tensor(ids[1:n + 1]), reduction="none")
    ce = ce_t.mean().item()
    print(f"float CE over the first 64 positions alone (causal, so a SEQ=64 run sees the same): {ce_t[:64].mean().item():.4f}")
    rng["resid"] = max(h.abs().max().item() for h in o.hidden_states)
for h in hooks: h.remove()
print(f"float GPT-2 mean next-token CE on the window: {ce:.4f}")
print(f"max |pre-GELU| {rng['pre_gelu']:.2f} (training table domain +-128)   max |residual| {rng['resid']:.1f}")
