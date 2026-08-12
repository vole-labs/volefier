# Three-way int8 comparison: fp32 (ORT) | float-scale int8 (ideal, exact requant scale) |
# ml-quant (ZK fixed-point c*2^e requant). The float-scale row is the baseline BETWEEN fp32 and ml-quant.
import sys, numpy as np, glob, csv, onnx, onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
_o=onnx.load; _c={}
def cl(p,*a,**k):
    if p not in _c:_c[p]=_o(p,*a,**k)
    return _c[p]
onnx.load=cl
G={'onnx':onnx}; exec(open('tools/mimics/plaintext_quant.py').read().split('if __name__')[0],G)   # run + FLOAT_SCALE in G
run_q=G['run']
syn2idx={r["class_id"]:int(r["id"]) for r in csv.DictReader(open("model/eval/imagenet/class_map.csv"))}
def prep(p):
    im=Image.open(p).convert("RGB"); w,h=im.size; s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR); w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
def q_argmax(MQ,xr,ds,dz,float_scale):
    G['FLOAT_SCALE']=float_scale
    t=xr/ds; xq=np.clip(np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5))+dz,0,255).astype(np.int64).reshape(1,3,224,224)
    return int(run_q(MQ,xq,ds,dz).argmax())
def sweep(name,MF,MQ,nimg):
    fp32=ort.InferenceSession(MF,providers=["CPUExecutionProvider"]); inn=fp32.get_inputs()[0].name
    qi={t.name:numpy_helper.to_array(t) for t in onnx.load(MQ).graph.initializer}; ds=float(qi["data_scale"]); dz=int(qi["data_zero_point"])
    samp=[(sy,sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))[0]) for sy in sorted(syn2idx)[::max(1,1000//nimg)][:nimg]]
    af=afl=aq=0; dfl=dq=0; n=0
    for sy,f in samp:
        x=prep(f); lab=syn2idx[sy]; xr=x.ravel(); n+=1
        fa=int(fp32.run(None,{inn:x})[0].argmax()); af+=fa==lab
        fl=q_argmax(MQ,xr,ds,dz,True);  afl+=fl==lab; dfl+=fl!=fa
        zq=q_argmax(MQ,xr,ds,dz,False); aq +=zq==lab; dq +=zq!=fa
    print(f"=== {name} ({n} imgs) ===")
    print(f"  {'backend':22s}  top-1     differs-from-fp32")
    print(f"  {'fp32 (ORT)':22s}  {100*af/n:5.1f}%    --")
    print(f"  {'float-scale int8':22s}  {100*afl/n:5.1f}%    {100*dfl/n:.1f}%")
    print(f"  {'ml-quant (c*2^e int8)':22s}  {100*aq/n:5.1f}%    {100*dq/n:.1f}%",flush=True)
sweep("ResNet-18","model/eval/resnet18-v1-7/resnet18-v1-7.onnx","model/resnet18-v1-7-qdq.onnx",100)
sweep("VGG-16","model/eval/vgg16/vgg16.onnx","model/vgg16-12-qdq.onnx",60)
