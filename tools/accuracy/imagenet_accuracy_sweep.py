import sys, numpy as np, glob, csv, onnx, onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
FP="model/eval/resnet50-v1-7/resnet50-v1-7.onnx"
QDQ="model/resnet50-v1-7-qdq-imagenet.onnx"
MODE=sys.argv[1] if len(sys.argv)>1 else "all"      # "fp" = fp32 only (fast), "all" = 3-way
N=int(sys.argv[2]) if len(sys.argv)>2 else 250
def prep(p):
    im=Image.open(p).convert("RGB");w,h=im.size;s=256/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BILINEAR);w,h=im.size
    im=im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224))
    return ((np.asarray(im,np.float32)/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
syn2idx={r["class_id"]:int(r["id"]) for r in csv.DictReader(open("model/eval/imagenet/class_map.csv"))}
syns=sorted(syn2idx)[::max(1,1000//N)][:N]
fp32=ort.InferenceSession(FP,providers=["CPUExecutionProvider"]);inn=fp32.get_inputs()[0].name
if MODE=="all":
    Gf={'onnx':onnx,'np':np};exec(open('tools/mimics/plaintext_fixed.py').read().split('if __name__')[0],Gf);run_fixed=Gf['run']
    Gq={'onnx':onnx,'np':np};exec(open('tools/mimics/plaintext_quant.py').read().split('if __name__')[0],Gq);run_quant=Gq['run']
    qi={t.name:numpy_helper.to_array(t) for t in onnx.load(QDQ).graph.initializer};ds=float(qi["data_scale"]);dz=int(qi["data_zero_point"])
afp=afix=aq=a8=agf=agq=ag8=n=0
fix_ne_quant=[]            # images where ml-fixed argmax != ml-quant argmax (DIRECT count)
fix_ne_w8=[]               # images where ml-fixed argmax != W8A8-ideal argmax (DIRECT count)
quant_ne_w8=[0]            # ml-quant-ZK argmax != W8A8-ideal argmax (pure ZK-requant divergence)
for sy in syns:
    f=sorted(glob.glob(f"model/eval/imagenet/val/{sy}/*.JPEG"))[0]
    x=prep(f);lab=syn2idx[sy];n+=1
    fa=int(fp32.run(None,{inn:x})[0].argmax());afp+=fa==lab
    if MODE=="all":
        xf=x.ravel().astype(np.float32)
        fix=int(run_fixed(FP,xf).argmax());afix+=fix==lab;agf+=fix==fa
        t=xf/ds;xq=np.clip(np.where(t>=0,np.floor(t+0.5),np.ceil(t-0.5))+dz,0,255).astype(np.int64).reshape(1,3,224,224)
        Gq['FLOAT_SCALE']=True;  q8=int(run_quant(QDQ,xq,ds,dz).argmax()); a8+=q8==lab; ag8+=q8==fa   # real W8A8: exact float scale
        Gq['FLOAT_SCALE']=False; qa=int(run_quant(QDQ,xq,ds,dz).argmax()); aq+=qa==lab; agq+=qa==fa   # ml-quant ZK: c*2^e requant
        if fix!=qa: fix_ne_quant.append((f.split('/')[-1],fix,qa))
        if fix!=q8: fix_ne_w8.append((f.split('/')[-1],fix,q8))
        if q8!=qa: quant_ne_w8[0]+=1
print(f"\n=== ResNet-50 / ImageNet, {n} images (1 per class) ===")
print(f"  fp32 (ORT, BASELINE)     top-1 = {afp/n:.1%}")
if MODE=="all":
    print(f"  W8A8 ideal (exact scale) top-1 = {a8/n:.1%}   agree-with-fp32 = {ag8/n:.1%}   <- int8-quant error only")
    print(f"  ml-quant ZK (c*2^e)      top-1 = {aq/n:.1%}   agree-with-fp32 = {agq/n:.1%}   <- + ZK-requant error")
    print(f"  ml-fixed (P=12)          top-1 = {afix/n:.1%}   agree-with-fp32 = {agf/n:.1%}")
    print(f"\n  ml-quant ZK vs W8A8-ideal DIFFER on {quant_ne_w8[0]} / {n}  (pure ZK-requant divergence)")
    print(f"  ml-fixed vs W8A8-ideal   DIFFER on {len(fix_ne_w8)} / {n}  (fp32 vs int8-quant, since ml-fixed==fp32)")
    for nm,fx,q8 in fix_ne_w8: print(f"    {nm}: ml-fixed={fx}  W8A8={q8}")
    print(f"  ml-fixed vs ml-quant     DIFFER on {len(fix_ne_quant)} / {n} images")
    for nm,fix,qa in fix_ne_quant: print(f"    {nm}: ml-fixed={fix}  ml-quant={qa}")
