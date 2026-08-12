# ml-fixed precision sweep: top-1 and agree-with-fp32 as a function of P (fixed-point bits).
# Usage: python3 tools/fixed_precision_sweep.py LABEL FP_ONNX N "8,10,12,14,16,18,20"
import sys, os, numpy as np, glob, csv, onnx, onnxruntime as ort
from PIL import Image
LABEL=sys.argv[1]; FP=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 100
Ps=[int(x) for x in (sys.argv[4] if len(sys.argv)>4 else "8,10,12,14,16,18,20").split(",")]
def prep(p):
    im=Image.open(p).convert("RGB");w,h=im.size;s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR);w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
syn2idx={r["class_id"]:int(r["id"]) for r in csv.DictReader(open("model/eval/imagenet/class_map.csv"))}
syns=sorted(syn2idx)[::max(1,1000//N)][:N]
fp32=ort.InferenceSession(FP,providers=["CPUExecutionProvider"]);inn=fp32.get_inputs()[0].name
# precompute fp32 argmax + label + flattened input once
data=[]; afp=0
for sy in syns:
    fs=sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))
    if not fs: continue
    x=prep(fs[0]); lab=syn2idx[sy]; fa=int(fp32.run(None,{inn:x})[0].argmax()); afp+=fa==lab
    data.append((x.ravel().astype(np.float32),fa,lab))
n=len(data)
src=open('tools/mimics/plaintext_fixed.py').read().split('if __name__')[0]
print(f"\n=== {LABEL} / ImageNet, {n} images (1 per class): ml-fixed precision sweep ===")
print(f"  fp32 baseline top-1 = {afp/n:.1%}")
print(f"  {'P':>3}  {'ml-fixed top-1':>14}  {'agree-with-fp32':>16}")
for p in Ps:
    os.environ["P"]=str(p)
    G={'onnx':onnx,'np':np,'os':os}; exec(src,G); run_fixed=G['run']
    top1=agree=0
    for xf,fa,lab in data:
        fix=int(run_fixed(FP,xf).argmax()); top1+=fix==lab; agree+=fix==fa
    print(f"  {p:>3}  {top1/n:>13.1%}  {agree/n:>15.1%}", flush=True)
