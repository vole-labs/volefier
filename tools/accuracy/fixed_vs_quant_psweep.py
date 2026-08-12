# Table-6 sweep: ACCURACY (top-1 vs label) AND top-1 agreement (fixed<->fp32, fixed<->quant),
# as a function of fixed-point precision P, plus the deployed int8 model (quant<->fp32, quant acc).
# Prints raw counts so runs on disjoint image sets can be combined.
# Usage: python3 tools/fixed_vs_quant_psweep.py LABEL FP QDQ N Ps OFFSET STRIDE IMG_LO IMG_HI
#   N-grid classes = sorted(syns)[::1000//N][:N][OFFSET::STRIDE]; per class use images [IMG_LO:IMG_HI].
import sys, os, numpy as np, glob, csv, onnx, onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
LABEL=sys.argv[1]; FP=sys.argv[2]; QDQ=sys.argv[3]; N=int(sys.argv[4]) if len(sys.argv)>4 else 100
Ps=[int(x) for x in (sys.argv[5] if len(sys.argv)>5 else "8,10,12,14").split(",")]
OFFSET=int(sys.argv[6]) if len(sys.argv)>6 else 0
STRIDE=int(sys.argv[7]) if len(sys.argv)>7 else 1
IMG_LO=int(sys.argv[8]) if len(sys.argv)>8 else 0
IMG_HI=int(sys.argv[9]) if len(sys.argv)>9 else 1
def prep(p):
    im=Image.open(p).convert("RGB");w,h=im.size;s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR);w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
syn2idx={r["class_id"]:int(r["id"]) for r in csv.DictReader(open("model/eval/imagenet/class_map.csv"))}
syns=sorted(syn2idx)[::max(1,1000//N)][:N][OFFSET::STRIDE]
fp32=ort.InferenceSession(FP,providers=["CPUExecutionProvider"]);inn=fp32.get_inputs()[0].name
Gq={'onnx':onnx,'np':np}; exec(open('tools/mimics/plaintext_quant.py').read().split('if __name__')[0],Gq); run_quant=Gq['run']
qi={t.name:numpy_helper.to_array(t) for t in onnx.load(QDQ).graph.initializer}; ds=float(qi["data_scale"]); dz=int(qi["data_zero_point"])
# precompute fp32 argmax, deployed-quant argmax, label, flattened input (once) over classes x image-range
data=[]; agq=0; afp=0; aqa=0
for sy in syns:
    fs=sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))[IMG_LO:IMG_HI]
    lab=syn2idx[sy]
    for fpath in fs:
        x=prep(fpath); xf=x.ravel().astype(np.float32)
        fa=int(fp32.run(None,{inn:x})[0].argmax()); afp+=fa==lab
        t=xf/ds; xq=np.clip(np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5))+dz,0,255).astype(np.int64).reshape(1,3,224,224)
        Gq['FLOAT_SCALE']=False; qa=int(run_quant(QDQ,xq,ds,dz).argmax()); agq+=qa==fa; aqa+=qa==lab
        data.append((xf,fa,qa,lab))
n=len(data)
src=open('tools/mimics/plaintext_fixed.py').read().split('if __name__')[0]
print(f"\n=== {LABEL} / ImageNet, {n} images ({len(syns)} classes x [{IMG_LO}:{IMG_HI}]) ===")
print(f"  fp32 accuracy = {afp}/{n} = {afp/n:.1%}")
print(f"  quant accuracy = {aqa}/{n} = {aqa/n:.1%}   quant<->fp32 = {agq}/{n} = {agq/n:.1%}")
print(f"  {'P':>3}  {'accuracy':>12}  {'fixed<->fp32':>14}  {'fixed<->quant':>15}")
for p in Ps:
    os.environ["P"]=str(p)
    G={'onnx':onnx,'np':np,'os':os}; exec(src,G); run_fixed=G['run']
    acc=af=aq=0
    for xf,fa,qa,lab in data:
        fix=int(run_fixed(FP,xf).argmax()); acc+=fix==lab; af+=fix==fa; aq+=fix==qa
    print(f"  {p:>3}  {acc}/{n} {acc/n:>6.1%}  {af}/{n} {af/n:>6.1%}  {aq}/{n} {aq/n:>6.1%}", flush=True)
