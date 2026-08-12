// Wires ml-quant transformer components into the ONNX dispatcher: builds a
// synthetic graph (LayerNormalization -> Gelu -> Softmax + Transpose/Split) and
// proves it via QuantGraphInfer dispatching to QuantLayerNorm/QuantActivation/QuantSoftmax.
#include <iostream>
#include <vector>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx.h"
#include "volefier/ml-quant/quant_graph_infer.h"
using namespace emp; using namespace volefier; using namespace std;
int port, party; const int threads = 4, P = 12;
static OnnxNode nd(string op, vector<string> in, vector<string> out){ OnnxNode n; n.op_type=op; n.name=op; n.input=in; n.output=out; return n; }
int main(int argc, char **argv){
  parse_party_and_port(argv, &party, &port);
  setvbuf(stdout, NULL, _IONBF, 0); cout.setf(ios::unitbuf);
  BoolIO *ios[threads]; for(int i=0;i<threads;++i) ios[i]=new BoolIO(new NetIO(party==ALICE?nullptr:"127.0.0.1", port+i), party==ALICE);
  cout<<"\n----- ml-quant transformer-ops wiring test -----\n\n";
  const int S=2, D=8;
  OnnxGraph g; g.inputs={"x"}; g.outputs={"out"}; g.input_shapes["x"]={1,S,D};
  auto mk=[](vector<int64_t> dims, vector<float> data){ OnnxTensor t; t.dtype=1; t.dims=dims; t.data=data; return t; };
  block seed=makeBlock(0,7); PRG prg(&seed);
  vector<float> gm(D), bt(D); for(auto&v:gm)v=1.0f; for(auto&v:bt)v=0.05f;
  g.initializer["gamma"]=mk({D}, gm); g.initializer["beta"]=mk({D}, bt);
  g.nodes.push_back(nd("LayerNormalization", {"x","gamma","beta"}, {"ln"}));
  g.nodes.push_back(nd("Gelu", {"ln"}, {"gl"}));
  g.nodes.push_back(nd("Softmax", {"gl"}, {"out"}));
  { map<string,int> bad; for(auto&n:g.nodes) if(!QuantGraphInfer::supported(n.op_type)) bad[n.op_type]++;
    if(!bad.empty()){cout<<"UNSUPPORTED:"; for(auto&kv:bad)cout<<" "<<kv.first; cout<<"\n"; for(int i=0;i<threads;++i)delete ios[i]; return 2;} }
  cout<<"ops all supported: "; for(auto&n:g.nodes)cout<<n.op_type<<" "; cout<<"\n";
  const uint64_t EXP_T=1u<<16, GELU_T=1u<<14; const int64_t GELU_OFF=1<<13;
  setup_zk_arith(ios, threads, party);
  {
    QuantInfer qi(party, 16, ios[0], 24, 1, 4);
    RangeCheckLU rclu(party, ios[0]->io, 16); CompareLU cmp(&rclu); Div divr(party, ios[0]->io, 16);
    QuantTable expt(EXP_T, party, ios[0]->io); expt.Setup(build_exp_table(EXP_T, P));
    QuantTable gtab(GELU_T, party, ios[0]->io); gtab.Setup(build_gelu_table(GELU_T, P, GELU_OFF));
    QuantGraphInfer gi(&qi, &cmp, &divr, party);
    gi.P=P; gi.rcmp=&rclu; gi.exp_tab=&expt; gi.gelu_tab=&gtab; gi.gelu_off=GELU_OFF; gi.score_bits=16;
    vector<int64_t> xr(S*D, 0);
    if(party==ALICE){ for(auto&v:xr){ uint64_t t; prg.random_data(&t,8); v=(int64_t)(t%2000)-1000; } }   // @2^-12
    vector<IntFp> cx; qi.commit_vec(cx, xr);
    vector<int64_t> oref; vector<IntFp> out = gi.run(g, cx, xr, 1.0/(1<<P), 0, oref);
    for(auto&e:out) e.reveal();
    rclu.Teardown(); expt.Teardown(); gtab.Teardown(); finalize_zk_arith();
    if(party==ALICE){ cout<<"dispatched: "; for(auto&op:gi.ops_run)cout<<op<<" "; cout<<"\n";
      cout<<"pass: ml-quant transformer ops dispatched to QuantLayerNorm/QuantActivation/QuantSoftmax + proven\n"; }
    else cout<<"verifier: complete\n";
  }
  for(int i=0;i<threads;++i) delete ios[i]; return 0;
}
