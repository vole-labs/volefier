# Export BERT-base (input-ids-only, fixed seq) to fp32 ONNX + static-QDQ int8 — closes the BERT
# provenance gap.  Reproduces:
#   model/bert_base_fp32_basic.onnx  (ml-fixed BERT: fp32, plain ops, encoder + Tanh pooler + 2-class head)
#   model/bert_base_qdq.onnx         (ml-quant BERT: QDQ int8)
# Usage:  python3 tools/export_bert.py [seq=16] [hf_id=bert-base-uncased]
#   override outputs via env BERT_FP32 / BERT_QDQ (used by tests to export to a temp name).
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede torch/onnxruntime (duplicate libomp on macOS)
import numpy as np, torch
from transformers import BertForSequenceClassification

SEQ  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
HF   = sys.argv[2] if len(sys.argv) > 2 else "bert-base-uncased"
FP32 = os.environ.get("BERT_FP32", "model/bert_base_fp32_basic.onnx")
QDQ  = os.environ.get("BERT_QDQ",  "model/bert_base_qdq.onnx")

# 1) fp32 ONNX, input_ids only (attention_mask=ones, token_type=zeros default -> folded away),
#    fixed batch=1 seq=SEQ.  Random-init 2-class head: the ZK tests are path tests, not accuracy.
m = BertForSequenceClassification.from_pretrained(HF, num_labels=2).eval()
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, input_ids): return s.m(input_ids=input_ids).logits
ids = torch.zeros(1, SEQ, dtype=torch.long)
torch.onnx.export(Wrap(m), (ids,), FP32, input_names=["input_ids"], output_names=["logits"],
                  opset_version=20, do_constant_folding=True, dynamo=False)   # opset 20: single fused Gelu op; legacy exporter (no onnxscript dep)
print("wrote", FP32)

# 2) static QDQ int8 (synthetic token-id calibration -> provable; use real text calib for accuracy work).
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
pre = FP32.replace(".onnx", "-pre.onnx"); quant_pre_process(FP32, pre, skip_symbolic_shape=False)
class Calib(CalibrationDataReader):
    def __init__(s, n=16, V=30522): s.it = iter([{"input_ids": np.random.randint(0, V, (1, SEQ), np.int64)} for _ in range(n)])
    def get_next(s): return next(s.it, None)
quantize_static(pre, QDQ, Calib(), quant_format=QuantFormat.QDQ, per_channel=False,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
os.remove(pre)
print("wrote", QDQ)
