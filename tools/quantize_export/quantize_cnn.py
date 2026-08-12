# Download an ONNX Model Zoo fp32 image classifier (or take a local path) and static-QDQ-quantize it to
# int8 -> the model/<name>-qdq.onnx the ml-quant CNN proofs load.  Synthetic-image calibration: fine for
# the ZK path test; use real ImageNet calib (see tools/accuracy_baselines.py) if you care about accuracy.
#   python3 tools/quantize_cnn.py <fp32-url-or-path> <dst-qdq.onnx>
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede onnxruntime (duplicate libomp on macOS)
import urllib.request, numpy as np, onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

src, dst = sys.argv[1], sys.argv[2]
fp32_keep = sys.argv[3] if len(sys.argv) > 3 else None     # where to keep the downloaded fp32 (e.g. model/eval/<name>/ for accuracy_baselines.py)
if src.startswith("http"):
    fp = fp32_keep or dst.replace(".onnx", "-fp32.onnx")
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    print("downloading", src, "->", fp); urllib.request.urlretrieve(src, fp); src = fp

m = onnx.load(src, load_external_data=False); inp = m.graph.input[0]
name = inp.name; dims = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
print(f"input '{name}' shape {dims}")
pre = dst.replace(".onnx", "-pre.onnx"); quant_pre_process(src, pre, skip_symbolic_shape=False)
class Calib(CalibrationDataReader):
    def __init__(s, n=8): s.it = iter([{name: np.random.rand(*dims).astype(np.float32)} for _ in range(n)])
    def get_next(s): return next(s.it, None)
quantize_static(pre, dst, Calib(), quant_format=QuantFormat.QDQ, per_channel=False,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
os.remove(pre)
print("wrote", dst)
