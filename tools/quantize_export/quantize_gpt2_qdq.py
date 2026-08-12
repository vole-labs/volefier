# Static-QDQ-quantize the clean fp32 12-layer GPT-2 (model/gpt2-12_nocache) with onnxruntime's
# static quantizer. Per-tensor int8 QDQ around MatMul/Gemm; LayerNorm/Softmax/Gelu stay fp32.
import os, sys, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

src = sys.argv[1] if len(sys.argv) > 1 else "model/gpt2-12_nocache/model.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "model/gpt2-12-qdq.onnx"
pre = src.replace(".onnx", "-pre.onnx")

print("pre-processing (shape inference + fold) ...")
quant_pre_process(src, pre, skip_symbolic_shape=False)

class GPT2Calib(CalibrationDataReader):
    def __init__(self, n=16, S=8, vocab=50257):
        rng = np.random.RandomState(0)
        self.data = []
        for _ in range(n):
            ids = rng.randint(0, vocab, (1, S)).astype(np.int64)
            self.data.append({"input_ids": ids,
                              "attention_mask": np.ones((1, S), np.int64),
                              "position_ids": np.arange(S, dtype=np.int64)[None, :]})
        self.it = iter(self.data)
    def get_next(self): return next(self.it, None)

print("static QDQ quantization ...")
quantize_static(
    pre, dst, GPT2Calib(),
    quant_format=QuantFormat.QDQ,
    per_channel=False,                        # per-tensor (matches ml-quant)
    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,   # uint8 acts (matches prover's [0,255] clamp), int8 weights
    op_types_to_quantize=["MatMul", "Gemm"],  # leave LayerNorm/Softmax/Gelu fp32
)
print("wrote", dst, os.path.getsize(dst) / 1e6, "MB")
