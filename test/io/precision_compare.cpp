// Host-only: run the same fp32 CNN forward in float64 (truth), float32 (=ml-float),
// fixed-point 2^-P (=ml-fixed), and int8 via the QDQ model (=ml-quant); report each
// backend's logit deviation from the float64 reference. (Proofs are exact, so each
// backend's arithmetic output equals its proven logits.)
#include <cstdio>
#include <cmath>
#include <map>
#include <vector>
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_calib.h"
using namespace volefier; using namespace std;
static double rnd(double v,int mode,int P){ if(mode==1) return (double)(float)v; if(mode==2){ double s=(double)(1LL<<P); return std::round(v*s)/s; } return v; }
static vector<double> forward(OnnxGraph g, int mode, int P){
  map<string,HT> V; for(auto&kv:g.initializer) V[kv.first]=ht_from_init(kv.second);
  for(auto&n:g.nodes){
    if(n.op_type=="Split"){ auto it=V.find(n.input[0]); if(it==V.end())continue; const HT&x=it->second; int nd=x.shape.size(); int axis=n.attr.count("axis")?(int)n.attr.at("axis").i:0; if(axis<0)axis+=nd; int no=n.output.size(); int64_t each=x.shape[axis]/max(1,no),outer=1; for(int i=0;i<axis;i++)outer*=x.shape[i]; int64_t inner=1; for(int i=axis+1;i<nd;i++)inner*=x.shape[i]; int64_t off=0; for(int o=0;o<no;o++){HT p;p.dt=x.dt;p.shape=x.shape;p.shape[axis]=each;p.v.resize(outer*each*inner); for(int64_t oo=0;oo<outer;oo++)for(int64_t a=0;a<each;a++)for(int64_t k=0;k<inner;k++)p.v[(oo*each+a)*inner+k]=x.v[(oo*x.shape[axis]+off+a)*inner+k]; off+=each; V[n.output[o]]=p;} continue; }
    fprintf(stderr,"%s|%s\n",n.op_type.c_str(),n.output.empty()?"":n.output[0].c_str()); HT res; bool ok=false; try{ok=calib_eval(n,V,res);}catch(...){ok=false;}
    if(ok&&!n.output.empty()){ for(auto&v:res.v) v=rnd(v,mode,P); V[n.output[0]]=res; }
    if(V.count(g.outputs[0])) break;   // stop once the logits are computed (skip trailing nodes)
  }
  return V[g.outputs[0]].v;
}
static void report(const char*nm, const vector<double>&a, const vector<double>&ref){
  double mab=0,sab=0,mr=0; size_t n=min(a.size(),ref.size());
  for(size_t i=0;i<n;i++){ double e=fabs(a[i]-ref[i]); mab=max(mab,e); sab+=e; mr=max(mr,e/(fabs(ref[i])+1e-9)); }
  printf("  %-26s max|err|=%.6g  mean|err|=%.6g  max-rel=%.4g\n", nm, mab, sab/max((size_t)1,n), mr);
}
int main(int c,char**v){ setvbuf(stdout,NULL,_IONBF,0);
  const char* fp32=v[1]; const char* qdq=v[2]; int P=c>3?atoi(v[3]):12;
  OnnxGraph gf=load_onnx(fp32);
  // find input DQ scale/zp in the QDQ model (the node consuming "data")
  OnnxGraph gq=load_onnx(qdq); double dsc=1; double dzp=0;
  for(auto&n:gq.nodes) if(n.op_type=="DequantizeLinear"&&n.input[0]=="data"){ dsc=gq.initializer.at(n.input[1]).data.size()?gq.initializer.at(n.input[1]).data[0]:gq.initializer.at(n.input[1]).i64[0]; auto&z=gq.initializer.at(n.input[2]); dzp=z.i64.size()?z.i64[0]:(z.data.size()?z.data[0]:0); break; }
  // deterministic input on the quant grid: Xq uint8 pattern, X = (Xq-zp)*scale
  auto&s=gf.input_shapes[gf.inputs[0]]; for(auto&d:s) if(d<=0)d=1; int64_t ne=1; for(auto d:s)ne*=d;
  vector<double> X(ne),Xq(ne); for(int64_t i=0;i<ne;i++){ double q=64+(double)(i%128); Xq[i]=q; X[i]=(q-dzp)*dsc; }
  OnnxTensor tf; tf.name="data"; tf.dtype=1; tf.dims=s; tf.data.assign(X.begin(),X.end()); gf.initializer["data"]=tf;
  OnnxTensor tq; tq.name="data"; tq.dtype=1; tq.dims=s; tq.data.assign(Xq.begin(),Xq.end()); gq.initializer["data"]=tq;
  printf("input: data_scale=%.6g data_zp=%.0f, %lld logits\n", dsc, dzp, (long long)ne);
  printf("running float64 (truth)...\n"); auto ref=forward(gf,0,P);
  printf("\nLogit deviation from float64 reference (out dim %zu):\n", ref.size());
  printf("running float32...\n");      report("ml-float (float32)",     forward(gf,1,P), ref);
  printf("running fixed 2^-%d...\n",P); report("ml-fixed (fixed 2^-P)",   forward(gf,2,P), ref);
  printf("running int8 QDQ...\n");      report("ml-quant (int8 QDQ)",     forward(gq,0,P), ref);
  return 0;}
