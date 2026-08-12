# Adds a STANDARD int8 baseline (ONNX Runtime running the QDQ model natively: float scales,
# standard requant/nonlinear) alongside our ZK-adapted int8 (plaintext_quant, c*2^e requant + LUTs).
# Reports, over the same N images:
#   fp32 acc; std-int8 acc + std<->fp32; zk-int8 acc + zk<->fp32 + zk<->std-int8 (ZK-vs-standard faithfulness);
#   per P: fixed acc, fixed<->fp32, fixed<->std-int8, fixed<->zk-int8.
# Usage: python3 tools/fixed_vs_stdquant_psweep.py LABEL FP QDQ N Ps IMG_LO IMG_HI
import sys, os, numpy as np, glob, csv, onnx, onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
LABEL=sys.argv[1]; FP=sys.argv[2]; QDQ=sys.argv[3]; N=int(sys.argv[4]) if len(sys.argv)>4 else 100
Ps=[int(x) for x in (sys.argv[5] if len(sys.argv)>5 else "8,10,12,14").split(",")]
IMG_LO=int(sys.argv[6]) if len(sys.argv)>6 else 0
IMG_HI=int(sys.argv[7]) if len(sys.argv)>7 else 3
def prep(p):
    im=Image.open(p).convert("RGB");w,h=im.size;s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR);w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
syn2idx={r["class_id"]:int(r["id"]) for r in csv.DictReader(open("model/eval/imagenet/class_map.csv"))}
syns=sorted(syn2idx)[::max(1,1000//N)][:N]
fp32=ort.InferenceSession(FP,providers=["CPUExecutionProvider"]);inn=fp32.get_inputs()[0].name
std=ort.InferenceSession(QDQ,providers=["CPUExecutionProvider"]);innq=std.get_inputs()[0].name  # standard int8
Gq={'onnx':onnx,'np':np}; exec(open('tools/mimics/plaintext_quant.py').read().split('if __name__')[0],Gq); run_quant=Gq['run']
qi={t.name:numpy_helper.to_array(t) for t in onnx.load(QDQ).graph.initializer}; ds=float(qi["data_scale"]); dz=int(qi["data_zero_point"])
data=[]; afp=asa=astd_fp=azk=azk_fp=azk_std=0
for sy in syns:
    lab=syn2idx[sy]
    for fpath in sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))[IMG_LO:IMG_HI]:
        x=prep(fpath); xf=x.ravel().astype(np.float32)
        fa=int(fp32.run(None,{inn:x})[0].argmax()); afp+=fa==lab
        sa=int(std.run(None,{innq:x.reshape(1,3,224,224)})[0].argmax()); asa+=sa==lab; astd_fp+=sa==fa
        t=xf/ds; xq=np.clip(np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5))+dz,0,255).astype(np.int64).reshape(1,3,224,224)
        Gq['FLOAT_SCALE']=False; qa=int(run_quant(QDQ,xq,ds,dz).argmax()); azk+=qa==lab; azk_fp+=qa==fa; azk_std+=qa==sa
        data.append((xf,fa,sa,qa,lab))
n=len(data)
src=open('tools/mimics/plaintext_fixed.py').read().split('if __name__')[0]
print(f"\n=== {LABEL} / ImageNet, {n} images ({len(syns)} classes x [{IMG_LO}:{IMG_HI}]) ===")
print(f"  fp32 accuracy                 = {afp}/{n} = {afp/n:.1%}")
print(f"  std-int8 (ORT/QDQ) accuracy   = {asa}/{n} = {asa/n:.1%}   std<->fp32 = {astd_fp/n:.1%}")
print(f"  zk-int8 (c*2^e)    accuracy   = {azk}/{n} = {azk/n:.1%}   zk<->fp32 = {azk_fp/n:.1%}   zk<->std-int8 = {azk_std/n:.1%}")
print(f"  {'P':>3}  {'acc':>7}  {'fix<->fp32':>10}  {'fix<->std':>10}  {'fix<->zk':>9}")
for p in Ps:
    os.environ["P"]=str(p)
    G={'onnx':onnx,'np':np,'os':os}; exec(src,G); run_fixed=G['run']
    acc=af=asd=az=0
    for xf,fa,sa,qa,lab in data:
        fix=int(run_fixed(FP,xf).argmax()); acc+=fix==lab; af+=fix==fa; asd+=fix==sa; az+=fix==qa
    print(f"  {p:>3}  {acc/n:>6.1%}  {af/n:>9.1%}  {asd/n:>9.1%}  {az/n:>8.1%}", flush=True)
