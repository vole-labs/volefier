# Static-QDQ int8 quantization of the LLM fp32 ONNX from export_llm_onnx.py, calibrated on REAL
# WikiText-2 text (fetch first: tools/fetch_models.sh wikitext). Per-tensor int8 QDQ around
# MatMul/Gemm; norms / softmax / activations stay fp32. Output feeds ml-quant; the fp32 source
# feeds ml-fixed. Real calibration matters -- synthetic/random calibration badly hurts accuracy
# (see the ResNet-50 lesson), so we tokenize actual WikiText-2 (also DeepProve's ppl dataset).
#
# Usage:  python tools/quantize_llm_qdq.py <src.onnx> <dst.onnx> <hf_id_for_tokenizer> [seq] [n]
#   e.g.  python tools/quantize_llm_qdq.py model/qwen2.5-0.5b/model.onnx model/qwen2.5-0.5b-qdq.onnx \
#            Qwen/Qwen2.5-0.5B-Instruct 64 32
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede native imports (libomp double-init)
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoTokenizer

src   = sys.argv[1] if len(sys.argv) > 1 else "model/llm/model.onnx"
dst   = sys.argv[2] if len(sys.argv) > 2 else "model/llm-qdq.onnx"
hf_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-0.5B-Instruct"
S     = int(sys.argv[4]) if len(sys.argv) > 4 else 64      # calibration sequence length
N     = int(sys.argv[5]) if len(sys.argv) > 5 else 32      # number of calibration sequences
WT    = "model/eval/gpt2_eval/wikitext2_test.txt"

tok = AutoTokenizer.from_pretrained(hf_id)

def calib_seqs():
    if os.path.exists(WT):
        ids = tok(open(WT, encoding="utf-8").read(), return_tensors="np").input_ids[0]
        step = max(1, (len(ids) - S) // max(1, N))
        for k in range(N):
            chunk = ids[k * step : k * step + S]
            if len(chunk) == S:
                yield chunk.astype(np.int64)[None, :]
    else:
        print(f"WARNING: {WT} missing -> random-token calibration (worse accuracy). "
              "Run: tools/fetch_models.sh wikitext")
        rng = np.random.RandomState(0)
        for _ in range(N):
            yield rng.randint(0, int(tok.vocab_size), (1, S)).astype(np.int64)

class WTCalib(CalibrationDataReader):
    def __init__(self):
        self.data = [{"input_ids": ids,
                      "attention_mask": np.ones_like(ids),
                      "position_ids": np.arange(ids.shape[1], dtype=np.int64)[None, :]}
                     for ids in calib_seqs()]
        print(f"calibration: {len(self.data)} sequences of len {S} "
              f"({'WikiText-2' if os.path.exists(WT) else 'RANDOM'})")
        self.it = iter(self.data)
    def get_next(self): return next(self.it, None)

pre = src.replace(".onnx", "-pre.onnx")
print("pre-processing (shape inference + fold) ...")
try:                                                        # LLMs exceed the 2GB protobuf limit ->
    quant_pre_process(src, pre, skip_symbolic_shape=False, # keep weights in an external .data file
                      save_as_external_data=True, all_tensors_to_one_file=True)
except TypeError:                                           # older onnxruntime without these args
    quant_pre_process(src, pre, skip_symbolic_shape=False)
print("static QDQ quantization ...")
quantize_static(
    pre, dst, WTCalib(),
    quant_format=QuantFormat.QDQ,
    per_channel=False,                                               # per-tensor (matches ml-quant)
    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,   # uint8 acts / int8 weights
    op_types_to_quantize=["MatMul", "Gemm"],                         # leave norms/softmax/act fp32
    use_external_data_format=True,                                   # LLM > 2GB protobuf limit
)
for f in (pre, pre + ".data"):                              # drop the ~1GB shape-inferred intermediate
    if os.path.exists(f): os.remove(f)
print("wrote", dst, os.path.getsize(dst) / 1e6, "MB")
