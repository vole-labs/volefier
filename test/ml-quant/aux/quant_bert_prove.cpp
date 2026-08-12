// Prove a real BERT-base (QDQ int8, opset-20 fused Gelu + Tanh pooler) forward through ml-quant.
// Exported input_ids-only (attention_mask=ones / token_type=zeros baked + folded), so the graph is
// pure encoder + pooler(Tanh) + classifier.  Exercises the new Tanh LogUp gadget end-to-end.
//   build: test_ml-quant_quant_bert_prove ;  run: ./run on it.   Env: P (gadget precision), IDS, MODEL.
#include <iostream>
#include <vector>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/io/onnx_fold.h"
#include "volefier/io/onnx_fuse.h"
#include "volefier/ml-quant/quant_graph_infer.h"
#include "test/utility.h"
using namespace emp; using namespace volefier; using namespace std;
int port, party; const int threads=getenv("ZK_THREADS")?atoi(getenv("ZK_THREADS")):4; const int P=getenv("P")?atoi(getenv("P")):12;
int main(int argc,char**argv){
  parse_party_and_port(argv,&party,&port); setvbuf(stdout,NULL,_IONBF,0);
  int seq = getenv("SEQ") ? atoi(getenv("SEQ")) : 16;   // must match the fixed-seq export
  const char* peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";   // BOB connects to ALICE; set PEER_IP=<prover IP> on BOB for distributed
  BoolIO*ios[threads]; for(int i=0;i<threads;++i) ios[i]=new BoolIO(new NetIO(party==ALICE?nullptr:peer_ip,port+i),party==ALICE);
  // BENCH=1 (or PHASES=1): per-phase wall-time + COMMUNICATION (the VOLE-ZK cost metric), including the
  // LogUp lookup-teardown the [TIMING] op-profiler defers out.  Prover-side (ALICE) figures.
  const bool BENCH=getenv("BENCH")||getenv("PHASES");
  auto comm=[&](){ uint64_t c=0; for(int i=0;i<threads;++i) c+=io_counter(ios[i]->io); return c; };
  auto _ck=clock_start(); uint64_t _cb=comm(); auto _pk=_ck; uint64_t _pb=_cb;   // _ck/_cb: per-phase; _pk/_pb: proving window
  auto LAP=[&](const char*nm){ if(BENCH&&party==ALICE){ double us=time_from(_ck); uint64_t b=comm()-_cb; fprintf(stderr,"[BENCH] %-18s %8.3f s  %8.2f MB\n",nm,us/1e6,b/1e6); } _ck=clock_start(); _cb=comm(); };
  const char* mp = getenv("MODEL") ? getenv("MODEL") : "model/bert_base_qdq.onnx";
  OnnxGraph g=load_onnx(mp);
  auto inj=[&](const char*nm,vector<int64_t>iv){ OnnxTensor t; t.name=nm; t.dtype=7; t.dims={1,(int64_t)iv.size()}; t.i64=iv; g.initializer[nm]=t; g.input_shapes[nm]={1,(int64_t)iv.size()}; };
  vector<int64_t> ids;
  if(getenv("IDS")){string s=getenv("IDS"),c;for(char ch:s+","){if(ch==','){if(!c.empty())ids.push_back(atoll(c.c_str()));c.clear();}else c+=ch;}seq=(int)ids.size();}
  else for(int i=0;i<seq;++i)ids.push_back(101+i*7);     // arbitrary token ids (random-init classifier -> path test, not accuracy)
  inj("input_ids",ids);
  FoldStats fs; g=fold_constants(g,fs); FuseStats us; g=fuse_patterns(g,us);
  printf("[%d] folded+fused: %zu nodes (seq=%d)\n", party, g.nodes.size(), seq);
  LAP("load+fold+fuse"); _pk=clock_start(); _pb=comm();   // proving window starts here
  { map<string,int>bad; for(auto&n:g.nodes) if(!QuantGraphInfer::supported(n.op_type))bad[n.op_type]++;
    if(!bad.empty()){printf("UNSUPPORTED:"); for(auto&kv:bad)printf(" %s",kv.first.c_str()); printf("\n"); return 2;} }
  setup_zk_arith(ios,threads,party);
  LAP("setup_zk_arith");
  {
    QuantInfer qi(party,P,ios[0],32,1,threads); RangeCheckLU rclu(party,ios[0]->io,16); CompareLU cmp(&rclu); Div divr(party,ios[0]->io,16);
    // tables AUTO-SIZE with P: exp domain [0,512), gelu & tanh domain +-128 (tanh saturates within +-4).
    const uint64_t ET=1ull<<(P+9), GT=1ull<<(P+8), TT=1ull<<(P+8); const int64_t GOFF=1LL<<(P+7), TOFF=1LL<<(P+7);
    QuantTable expt(ET,party,ios[0]->io); expt.Setup(build_exp_table(ET,P,!getenv("FUSED_NL")));
    QuantTable gtab(GT,party,ios[0]->io); gtab.Setup(build_gelu_table(GT,P,GOFF));
    QuantTable ttab(TT,party,ios[0]->io); ttab.Setup(build_tanh_table(TT,P,TOFF));    // <-- the new Tanh pooler gadget
    // OPTIMIZATION: QuantizeLinear's [0,255] saturation as ONE LogUp lookup instead of TWO is_le
    // committed comparisons (DeepProve "saturation-as-a-lookup"). BERT pre-clamp range ~[-3.5k,6.5k]
    // -> domain +-16384 covers it. Default ON (NO_CLAMP_LUT to A/B against the comparison path).
    const uint64_t CT=1ull<<15; const int64_t COFF=1LL<<14;
    QuantTable clampt(CT,party,ios[0]->io); if(!getenv("NO_CLAMP_LUT")) clampt.Setup(build_clamp_table(CT,COFF,0,255));
    QuantGraphInfer gi(&qi,&cmp,&divr,party); gi.P=P; gi.rcmp=&rclu; gi.exp_tab=&expt; gi.gelu_tab=&gtab; gi.gelu_off=GOFF;
    gi.tanh_tab=&ttab; gi.tanh_off=TOFF; gi.score_bits=(size_t)(P+9); gi.uniform_requant=!getenv("NO_UNIFORM"); if(getenv("PUBLIC_WEIGHTS")) gi.commit_weights=false;
    if(!getenv("NO_CLAMP_LUT")){ gi.clamp_tab=&clampt; gi.clamp_off=COFF; }
    // DeepProve-style scale-baked FUSED tanh: trace each Tanh's producer DequantizeLinear (s_in,zp_in) and
    // consumer QuantizeLinear (s_out,zp_out), build a per-op uint8 table (dequant+tanh+requant+clamp).
    std::vector<QuantTable*> fused_tabs;
    if(getenv("FUSED_NL")){
      auto initf=[&](const string&nm)->double{ if(!g.initializer.count(nm))return 0.0; auto&t=g.initializer.at(nm); return t.dtype==1?(t.data.empty()?0.0:(double)t.data[0]):(t.i64.empty()?0.0:(double)t.i64[0]); };
      auto initi=[&](const string&nm)->int{ if(!g.initializer.count(nm))return 0; auto&t=g.initializer.at(nm); return t.i64.empty()?0:(int)t.i64[0]; };
      for(auto&n:g.nodes) if(n.op_type=="Tanh"){
        double s_in=0,s_out=0; int zp_in=0,zp_out=0;
        for(auto&m:g.nodes){
          if(m.op_type=="DequantizeLinear"&&!m.output.empty()&&m.output[0]==n.input[0]&&m.input.size()>=3){ s_in=initf(m.input[1]); zp_in=initi(m.input[2]); }
          if(m.op_type=="QuantizeLinear"&&!m.input.empty()&&m.input[0]==n.output[0]&&m.input.size()>=3){ s_out=initf(m.input[1]); zp_out=initi(m.input[2]); } }
        if(s_in>0&&s_out>0){
          QuantTable* t=new QuantTable(256,party,ios[0]->io); t->Setup(build_tanh_table_fused(256,s_in,(double)zp_in,s_out,(double)zp_out,0,255));
          gi.tanh_tabs_[n.output[0]]=t; gi.tanh_scales_[n.output[0]]={s_in,(double)zp_in,s_out,(double)zp_out}; gi.tanh_fused_off=0; fused_tabs.push_back(t);
          printf("[%d] built DeepProve scale-baked tanh table (s_in=%.5f zp_in=%d s_out=%.5f zp_out=%d)\n",party,s_in,zp_in,s_out,zp_out); } }
      int ng=0;                                            // BERT gelu: output feeds Reshape (not a direct Q) -> trace through shape ops
      for(auto&n:g.nodes) if(n.op_type=="Gelu"||n.op_type=="Erf"){
        double s_in=0,s_out=0; int zp_in=0,zp_out=0;
        for(auto&m:g.nodes) if(m.op_type=="DequantizeLinear"&&!m.output.empty()&&m.output[0]==n.input[0]&&m.input.size()>=3){ s_in=initf(m.input[1]); zp_in=initi(m.input[2]); }
        std::string cur=n.output[0];
        for(int guard=0; guard<12 && s_out<=0; ++guard){ std::string nxt;
          for(auto&m:g.nodes) if(!m.input.empty()&&m.input[0]==cur){
            if(m.op_type=="QuantizeLinear"&&m.input.size()>=3){ s_out=initf(m.input[1]); zp_out=initi(m.input[2]); break; }
            if(m.op_type=="Reshape"||m.op_type=="Transpose"||m.op_type=="Flatten"||m.op_type=="Identity") nxt=m.output[0]; }
          if(nxt.empty()) break; cur=nxt; }
        if(s_in>0&&s_out>0){
          QuantTable* t=new QuantTable(256,party,ios[0]->io); t->Setup(build_gelu_table_fused_asym(256,s_in,(double)zp_in,s_out,(double)zp_out,0,255));
          gi.gelu_asym_tabs_[n.output[0]]=t; gi.gelu_asym_scales_[n.output[0]]={s_in,(double)zp_in,s_out,(double)zp_out}; fused_tabs.push_back(t); ++ng; } }
      printf("[%d] built %d DeepProve scale-baked gelu tables (asymmetric uint8, fused)\n",party,ng);
    }
    vector<int64_t> xr(seq,0); vector<IntFp> cx; qi.commit_vec(cx,xr); vector<int64_t> oref;
    LAP("table-setup");
    printf("[%d] running %zu ops...\n", party, g.nodes.size());
    vector<IntFp> out=gi.run(g,cx,xr,1.0/(1<<P),0,oref);
    LAP("forward-prove");
    printf("[%d] dispatched %zu ops, %zu output logits\n", party, gi.ops_run.size(), out.size());
    // classification head: argmax over the (few) class logits (skip with NO_ARGMAX to isolate forward soundness)
    int cls=-1; if(!getenv("NO_ARGMAX")){ IntFp idx=cmp.argmax(out,24); cls=(int)idx.reveal(); } else for(auto&v:out) v.reveal();
    LAP("argmax");
    rclu.Teardown(); expt.Teardown(); gtab.Teardown(); ttab.Teardown(); for(auto t:fused_tabs)t->Teardown(); if(!getenv("NO_CLAMP_LUT")) clampt.Teardown(); LAP("lookup-teardown"); finalize_zk_arith(); LAP("finalize_zk_arith");
    if(BENCH&&party==ALICE){ double us=time_from(_pk); uint64_t b=comm()-_pb;
      fprintf(stderr,"[BENCH] === PROVING TOTAL   %8.3f s  %8.2f MB  (setup_zk -> finalize, excl. model load) ===\n",us/1e6,b/1e6);
      fprintf(stderr,"[BENCH] peak RSS  %.0f MB\n",getPeakRSS()/1e6); }
    for(auto t:fused_tabs) delete t;
    if(party==ALICE){
      double mn=1e18,mx=-1e18; for(size_t i=0;i<oref.size();++i){double v=oref[i]/(double)(1<<P); mn=min(mn,v);mx=max(mx,v);}
      printf("PROVED bert-base (QDQ int8): %zu ops, %zu class logits, range [%.3f,%.3f], argmax class=%d\n", gi.ops_run.size(), out.size(), mn, mx, cls);
      printf("pass: ml-quant proved real BERT-base forward incl. Tanh pooler (verifier accepted)\n");
    } else printf("verifier: bert proof complete\n");
  }
  for(int i=0;i<threads;++i)delete ios[i]; return 0;
}
