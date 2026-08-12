#include <cstdio>
#include <map>
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-quant/quant_graph_infer.h"
using namespace volefier; using namespace std;
int main(int c, char**v){
  OnnxGraph g=load_onnx(v[1]); int seq=8;
  auto inj=[&](const char*nm,int dt,vector<int64_t>dims,vector<int64_t>iv,vector<float>fv){ OnnxTensor t; t.name=nm; t.dtype=dt; t.dims=dims; t.i64=iv; t.data=fv; g.initializer[nm]=t; g.input_shapes[nm]=dims; };
  vector<int64_t> ids; for(int i=0;i<seq;++i) ids.push_back(i+5); inj("input_ids",7,{1,seq},ids,{});
  vector<int64_t> mask(seq,1); inj("attention_mask",7,{1,seq},mask,{});
  vector<int64_t> pos; for(int i=0;i<seq;++i) pos.push_back(i); inj("position_ids",7,{1,seq},pos,{});
  FoldStats fs; g=fold_constants(g,fs); FuseStats us; g=fuse_patterns(g,us);
  map<string,int> bad; for(auto&n:g.nodes) if(!QuantGraphInfer::supported(n.op_type)) bad[n.op_type]++;
  printf("after fold+fuse: %zu nodes\n", g.nodes.size());
  if(bad.empty()) printf("ALL OPS SUPPORTED by ml-quant -- GPT-2 graph fully dispatches\n");
  else { printf("unsupported:"); int t=0; for(auto&kv:bad){printf(" %s x%d",kv.first.c_str(),kv.second);t+=kv.second;} printf(" (%d)\n",t); }
  return 0;}
