#include <cstdio>
#include <algorithm>
#include <vector>
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/io/onnx_calib.h"
using namespace volefier; using namespace std;
int main(int c,char**v){
  OnnxGraph g=load_onnx(v[1]); int seq=c>2?atoi(v[2]):6;
  auto inj=[&](const char*nm,vector<int64_t>iv){OnnxTensor t;t.name=nm;t.dtype=7;t.dims={1,(int64_t)iv.size()};t.i64=iv;g.initializer[nm]=t;g.input_shapes[nm]={1,(int64_t)iv.size()};};
  vector<int64_t> ids,mask,pos; for(int i=0;i<seq;++i){ids.push_back(100+i);mask.push_back(1);pos.push_back(i);}
  inj("input_ids",ids); inj("attention_mask",mask); inj("position_ids",pos);
  FoldStats fs; g=fold_constants(g,fs); FuseStats us; g=fuse_patterns(g,us);
  auto rng=calibrate_ranges(g);
  // overall + per-op-type peak
  double gmax=0; for(auto&kv:rng) gmax=max(gmax,kv.second);
  printf("calibrated %zu tensors (seq=%d), global max-abs=%.3f\n", rng.size(), seq, gmax);
  // show the activation ranges by the op that PRODUCED them
  map<string,double> byop; for(auto&n:g.nodes) if(!n.output.empty()&&rng.count(n.output[0])) byop[n.op_type]=max(byop[n.op_type],rng[n.output[0]]);
  printf("peak activation magnitude by producing op:\n");
  for(auto&kv:byop) printf("  %-20s %.3f\n", kv.first.c_str(), kv.second);
  // how many distinct scales we'd need vs a single global P
  int over2=0,over16=0; for(auto&kv:byop){ if(kv.second>2) over2++; if(kv.second>16) over16++; }
  printf("op-types exceeding GELU table window(+-2.0): %d ; exceeding exp window(16): %d\n", over2, over16);
  int miss=0; for(auto&n:g.nodes){ if(n.output.empty())continue; if(!rng.count(n.output[0])){ if(++miss<=6) printf("UNCOMPUTED: %s -> %s\n", n.op_type.c_str(), n.output[0].c_str()); } } printf("total uncomputed-output nodes: %d of %zu\n", miss, g.nodes.size());
  return 0;}
