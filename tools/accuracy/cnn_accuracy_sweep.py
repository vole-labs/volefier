# Generalized 3-way ImageNet top-1 / agree-with-fp32 sweep (fp32 vs ml-fixed vs ml-quant).
# Usage: python3 tools/cnn_accuracy_sweep.py LABEL FP_ONNX QDQ_ONNX N
# Mimics reproduce the ZK prover integer arithmetic bit-for-bit (mimic acc = proof acc).
import sys, numpy as np, glob, csv, onnx, onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
LABEL=sys.argv[1]; FP=sys.argv[2]; QDQ=sys.argv[3]; N=int(sys.argv[4]) if len(sys.argv)>4 else 150
def prep(p):
    im=Image.open(p).convert("RGB");w,h=im.size;s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR);w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
syn2idx={r["class_id"]:int(r["id"]) for r in csv.DictReader(open("model/eval/imagenet/class_map.csv"))}
syns=sorted(syn2idx)[::max(1,1000//N)][:N]
fp32=ort.InferenceSession(FP,providers=["CPUExecutionProvider"]);inn=fp32.get_inputs()[0].name
Gf={'onnx':onnx,'np':np};exec(open('tools/mimics/plaintext_fixed.py').read().split('if __name__')[0],Gf);run_fixed=Gf['run']
Gq={'onnx':onnx,'np':np};exec(open('tools/mimics/plaintext_quant.py').read().split('if __name__')[0],Gq);run_quant=Gq['run']
qi={t.name:numpy_helper.to_array(t) for t in onnx.load(QDQ).graph.initializer};ds=float(qi["data_scale"]);dz=int(qi["data_zero_point"])
afp=afix=aq=a8=agf=agq=ag8=n=0; fix_ne_quant=0
for sy in syns:
    fs=sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))
    if not fs: continue
    f=fs[0]; x=prep(f); lab=syn2idx[sy]; n+=1
    fa=int(fp32.run(None,{inn:x})[0].argmax()); afp+=fa==lab
    xf=x.ravel().astype(np.float32)
    fix=int(run_fixed(FP,xf).argmax()); afix+=fix==lab; agf+=fix==fa
    t=xf/ds; xq=np.clip(np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5))+dz,0,255).astype(np.int64).reshape(1,3,224,224)
    Gq['FLOAT_SCALE']=True;  q8=int(run_quant(QDQ,xq,ds,dz).argmax()); a8+=q8==lab; ag8+=q8==fa
    Gq['FLOAT_SCALE']=False; qa=int(run_quant(QDQ,xq,ds,dz).argmax()); aq+=qa==lab; agq+=qa==fa
    if fix!=qa: fix_ne_quant+=1
print(f"\n=== {LABEL} / ImageNet, {n} images (1 per class) ===")
print(f"  fp32 (ORT, BASELINE)     top-1 = {afp/n:.1%}")
print(f"  W8A8 ideal (exact scale) top-1 = {a8/n:.1%}   agree-with-fp32 = {ag8/n:.1%}")
print(f"  ml-quant ZK (c*2^e)      top-1 = {aq/n:.1%}   agree-with-fp32 = {agq/n:.1%}")
print(f"  ml-fixed (P=12)          top-1 = {afix/n:.1%}   agree-with-fp32 = {agf/n:.1%}")
print(f"  ml-fixed vs ml-quant DIFFER on {fix_ne_quant} / {n}")
