#include <iostream>
#include <map>
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
using namespace volefier; using namespace std;
static string SUP=" Gemm MatMul Conv BatchNormalization LayerNormalization Relu Sigmoid Gelu Erf Softmax MaxPool AveragePool GlobalAveragePool Add Mul Flatten Reshape Dropout Identity Gather Transpose Split ";
static void report(const char* tag, const OnnxGraph& g){
  map<string,int> h; int notsup=0; for(auto&n:g.nodes) h[n.op_type]++;
  for(auto&kv:h) if(SUP.find(" "+kv.first+" ")==string::npos) notsup+=kv.second;
  cout<<tag<<": "<<g.nodes.size()<<" nodes, "<<notsup<<" unsupported\n";
}
int main(int c, char** v){ setvbuf(stdout,NULL,_IONBF,0);
  OnnxGraph g=load_onnx(v[1]);
  for(auto&nm:g.inputs) if(!g.initializer.count(nm)){auto&s=g.input_shapes[nm]; for(size_t i=0;i<s.size();++i) if(s[i]<=0)s[i]=(i==0?1:8);}
  report("raw   ", g);
  FoldStats fs; OnnxGraph folded=fold_constants(g, fs); report("folded", folded);
  FuseStats us; OnnxGraph fused=fuse_patterns(folded, us);
  cout<<"fused "<<us.layernorm<<" LayerNorm + "<<us.gelu<<" GELU\n";
  report("fused ", fused);
  cout<<"\nremaining ops:\n"; map<string,int> h; for(auto&n:fused.nodes) h[n.op_type]++;
  for(auto&kv:h) cout<<"  "<<kv.second<<"\t"<<kv.first<<(SUP.find(" "+kv.first+" ")==string::npos?"   <-- NOT supported":"")<<"\n";
  return 0;
}
