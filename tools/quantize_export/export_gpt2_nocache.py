# Export GPT-2 small (12-layer) with use_cache=False to a no-cache fp32 ONNX,
# matching model/gpt2-12_nocache/ I/O exactly: inputs input_ids/attention_mask/position_ids,
# output logits[batch, seq, 50257], opset 18.
import os, sys, torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from transformers import GPT2LMHeadModel

name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"           # 12-layer GPT-2 small
outdir = sys.argv[2] if len(sys.argv) > 2 else "model/gpt2-12_nocache"
os.makedirs(outdir, exist_ok=True)

model = GPT2LMHeadModel.from_pretrained(name, use_cache=False, attn_implementation="eager")
model.eval()
cfg = model.config
print(f"{name}: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} vocab={cfg.vocab_size}")

class NoCacheWrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, position_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, use_cache=False).logits

wrap = NoCacheWrap(model).eval()
S = 8
ids = torch.randint(0, cfg.vocab_size, (1, S), dtype=torch.long)
mask = torch.ones(1, S, dtype=torch.long)
pos = torch.arange(S, dtype=torch.long).unsqueeze(0)

path = os.path.join(outdir, "model.onnx")
torch.onnx.export(
    wrap, (ids, mask, pos), path,
    input_names=["input_ids", "attention_mask", "position_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                  "attention_mask": {0: "batch_size", 1: "sequence_length"},
                  "position_ids": {0: "batch_size", 1: "sequence_length"},
                  "logits": {0: "batch_size", 1: "sequence_length"}},
    opset_version=18, do_constant_folding=True, dynamo=False)
print("wrote", path, os.path.getsize(path) / 1e6, "MB")
