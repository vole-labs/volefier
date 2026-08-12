# Export an HF causal-LM (Qwen / Llama / Gemma / GPT-2) to a NO-CACHE fp32 ONNX, matching the
# I/O convention of model/gpt2-12_nocache: inputs input_ids/attention_mask/position_ids, output
# logits[batch, seq, vocab], opset 18. This fp32 graph feeds ml-fixed directly and is the source
# for the QDQ int8 (ml-quant) via tools/quantize_llm_qdq.py.
#
# Usage:  python tools/export_llm_onnx.py <hf_id> <outdir>
#   e.g.  python tools/export_llm_onnx.py Qwen/Qwen2.5-0.5B-Instruct model/qwen2.5-0.5b
#
# Needs torch + a recent transformers (Qwen2.5 >= 4.37, Llama-3.2 >= 4.45, Gemma-3 >= 4.50).
# Gated repos (google/gemma-*, meta-llama/*) require `huggingface-cli login` + license acceptance;
# open alternatives: Qwen/Qwen2.5-0.5B-Instruct, TinyLlama/TinyLlama-1.1B-Chat-v1.0, unsloth/Llama-3.2-1B.
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede torch import (libomp double-init)
import torch
from transformers import AutoModelForCausalLM

hf_id  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
outdir = sys.argv[2] if len(sys.argv) > 2 else "model/llm"
# FIXED_SEQ>0 exports at a fixed sequence length with NO dynamic axes, so the shape/mask
# machinery folds to constants (5629 -> ~1500 nodes on Qwen-0.5B) and the graph becomes
# dispatchable by the ml-quant/ml-fixed front end. 0 (default) = dynamic axes.
FIXED_SEQ = int(os.environ.get("FIXED_SEQ", "0"))
os.makedirs(outdir, exist_ok=True)

def _load(**kw):
    return AutoModelForCausalLM.from_pretrained(
        hf_id, use_cache=False, attn_implementation="eager", **kw)
try:
    try:
        model = _load(dtype=torch.float32)          # transformers >= 5
    except TypeError:
        model = _load(torch_dtype=torch.float32)    # transformers 4.x
except Exception as e:
    sys.exit(f"\nfailed to load '{hf_id}': {e}\n"
             "If this is a gated repo (gemma/llama), accept its license on huggingface.co and run "
             "`huggingface-cli login`, or use an open mirror (see header).")
model.eval()
cfg = model.config
vocab = int(cfg.vocab_size)
print(f"{hf_id}: layers={getattr(cfg,'num_hidden_layers','?')} hidden={getattr(cfg,'hidden_size','?')} "
      f"heads={getattr(cfg,'num_attention_heads','?')} kv_heads={getattr(cfg,'num_key_value_heads','?')} "
      f"vocab={vocab}")

class NoCacheWrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, position_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, use_cache=False).logits

wrap = NoCacheWrap(model).eval()
S = FIXED_SEQ if FIXED_SEQ > 0 else 8
ids  = torch.randint(0, vocab, (1, S), dtype=torch.long)
mask = torch.ones(1, S, dtype=torch.long)
pos  = torch.arange(S, dtype=torch.long).unsqueeze(0)

# dynamic axes only when not pinning a fixed sequence length
dyn = None if FIXED_SEQ > 0 else {
    "input_ids":      {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "position_ids":   {0: "batch_size", 1: "sequence_length"},
    "logits":         {0: "batch_size", 1: "sequence_length"}}

path = os.path.join(outdir, "model.onnx")
with torch.no_grad():
    torch.onnx.export(
        wrap, (ids, mask, pos), path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"], dynamic_axes=dyn,
        opset_version=18, do_constant_folding=True, dynamo=False)
print("wrote", path, os.path.getsize(path) / 1e6, "MB",
      f"({'fixed seq='+str(S) if FIXED_SEQ>0 else 'dynamic'}; + external weights if >2GB)")
