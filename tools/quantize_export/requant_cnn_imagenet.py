import os, sys, glob, numpy as np, onnx
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
SRC="model/eval/resnet50-v1-7/resnet50-v1-7.onnx"
DST="model/resnet50-v1-7-qdq-imagenet.onnx"
NCAL=int(sys.argv[1]) if len(sys.argv)>1 else 100
inn=onnx.load(SRC).graph.input[0].name
def prep(p):
    im=Image.open(p).convert("RGB"); w,h=im.size; s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR); w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
# spread calibration across classes (every 1000//NCAL-th synset), 1 img each
syns=sorted(os.path.basename(d) for d in glob.glob("model/eval/imagenet/val/n*"))
imgs=[sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))[0] for sy in syns[::max(1,1000//NCAL)][:NCAL]]
class C(CalibrationDataReader):
    def __init__(s): s.it=iter([{inn:prep(p)} for p in imgs])
    def get_next(s): return next(s.it,None)
pre=DST.replace(".onnx","-pre.onnx"); quant_pre_process(SRC,pre,skip_symbolic_shape=False)
quantize_static(pre,DST,C(),quant_format=QuantFormat.QDQ,per_channel=False,
                activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8)
os.remove(pre)
qi={t.name:onnx.numpy_helper.to_array(t) for t in onnx.load(DST).graph.initializer}
print("wrote",DST,"| data_scale=",float(qi["data_scale"]),"zp=",int(qi["data_zero_point"]),"| calibrated on",len(imgs),"real imagenet imgs")
