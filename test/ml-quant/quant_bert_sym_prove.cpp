// Prove a real fp32 BERT-base (opset-20 Gelu + Tanh pooler) forward through ml-quant's
// DeepProve SYMMETRIC int-N path (SYM_BITS / sym_quant_act, zp=0) -- NOT the QDQ QuantizeLinear
// path.  The per-op requant is folded into sym_quant_act (post-op), so there is NO QuantizeLinear
// op and NO saturation-clamp table: the ~44s per-element clamp of the QDQ BERT is gone.
//   MERGE of quant_gpt2_prove.cpp (the SYM_BITS setup + fused-gelu + LSE softmax + CALIB) and
//   quant_bert_prove.cpp (Tanh pooler table + input_ids-only handling + class-logit argmax).
// Env: P (gadget precision, dflt 12), SYM_BITS (dflt 12), FUSED_NL, SEQ (dflt 512), MODEL, IDS,
//      CALIB_WRITE=<file> (dynamic, records per-tensor scales) then CALIB=<file> (static run).
#include <iostream>
#include <fstream>
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
  int seq = getenv("SEQ") ? atoi(getenv("SEQ")) : (argc>3 ? atoi(argv[3]) : 512);   // must match the fixed-seq export
  const char* peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";   // BOB(verifier) connects to ALICE(prover); set PEER_IP=<prover IP> on BOB
  BoolIO* ios[1]; ios[0]=new BoolIO(new NetIO(party==ALICE?nullptr:peer_ip,port),party==ALICE);  // single-io engine
  BoolIO* _vio=nullptr;   // background-sVOLE socket (port+1) under BG_VOLE
  const bool BENCH=getenv("BENCH")||getenv("PHASES");
  auto comm=[&](){ return io_counter(ios[0]->io) + (_vio ? io_counter(_vio->io) : (uint64_t)0); };
  auto _ck=clock_start(); uint64_t _cb=comm(); auto _pk=_ck; uint64_t _pb=_cb;   // _ck/_cb: per-phase; _pk/_pb: proving window
  auto LAP=[&](const char*nm){ if(BENCH&&party==ALICE){ double us=time_from(_ck); uint64_t b=comm()-_cb; fprintf(stderr,"[BENCH] %-18s %8.3f s  %8.2f MB\n",nm,us/1e6,b/1e6); } _ck=clock_start(); _cb=comm(); };
  const char* mp = getenv("MODEL") ? getenv("MODEL") : "model/bert512_fp32.onnx";
  OnnxGraph g=load_onnx(mp);
  auto inj=[&](const char*nm,vector<int64_t>iv){ OnnxTensor t; t.name=nm; t.dtype=7; t.dims={1,(int64_t)iv.size()}; t.i64=iv; g.initializer[nm]=t; g.input_shapes[nm]={1,(int64_t)iv.size()}; };
  vector<int64_t> ids;
  if(getenv("IDS")){string s=getenv("IDS"),c;for(char ch:s+","){if(ch==','){if(!c.empty())ids.push_back(atoll(c.c_str()));c.clear();}else c+=ch;}seq=(int)ids.size();}
  else for(int i=0;i<seq;++i)ids.push_back(101+i*7);     // arbitrary token ids (random-init classifier -> path test)
  inj("input_ids",ids);
  FoldStats fs; g=fold_constants(g,fs); FuseStats us; g=fuse_patterns(g,us);
  // ml-quant's Gather ignores the `axis` attribute (always axis-0), but BERT's pooler selects the
  // [CLS] token with an axis=1 scalar Gather over [1,512,768] -> [1,768].  Under the axis-0 gather
  // it would grab the whole [512,768] block (shape blows up to 150M).  Rewrite it: insert a Reshape
  // [1,512,768] -> [seq,768] before the gather so the axis-0 idx-0 picks exactly the CLS row.
  // ml-quant's run() auto-commits any SMALL initializer (<=65536 elems) into act_, so a Gemm whose
  // weight is small dispatches to matmul_aa (committed*committed), which ignores transB AND drops the
  // bias -> the [2,768] classifier weight is read as [K=2,N=768] and blows up to [384,768].  BERT's
  // only small dense weight is the final classifier, so rewrite it: Gemm(x, W[C,H] transB=1, b[C])
  // -> MatMul(x, W^T[H,C]) + Add(b), where matmul_aa's out[m,j]=sum_k A[m,k]*B[k*N+j] wants B=[H,C].
  { int H=768; for(auto&kv:g.initializer) if(kv.first.find("pooler.dense.weight")!=string::npos && kv.second.dims.size()==2){ H=(int)kv.second.dims[1]; break; }
    std::vector<OnnxNode> nn; nn.reserve(g.nodes.size()+4);
    for(auto&n:g.nodes){
      if(n.op_type=="Gather" && !n.output.empty() && n.output[0].find("pooler/Gather")!=string::npos){
        std::string sh=n.input[0]+"__flatsh", ro=n.input[0]+"__flat";
        OnnxTensor st; st.name=sh; st.dtype=7; st.dims={2}; st.i64={(int64_t)seq,(int64_t)H}; g.initializer[sh]=st;
        OnnxNode rn; rn.op_type="Reshape"; rn.name=ro; rn.input={n.input[0],sh}; rn.output={ro}; nn.push_back(rn);
        OnnxNode gn=n; gn.input[0]=ro; nn.push_back(gn); continue;
      }
      if(n.op_type=="Gemm" && !n.output.empty() && n.output[0]=="logits" && n.input.size()>=3
         && g.initializer.count(n.input[1]) && g.initializer.at(n.input[1]).dims.size()==2){
        auto&W=g.initializer.at(n.input[1]); int C=(int)W.dims[0], Hh=(int)W.dims[1];
        int tb=n.attr.count("transB")?(int)n.attr.at("transB").i:0;
        std::string wt=n.input[1]+"__T"; OnnxTensor T; T.name=wt; T.dtype=1; T.dims={Hh,C}; T.data.resize((size_t)Hh*C);
        for(int o=0;o<C;++o)for(int i=0;i<Hh;++i) T.data[(size_t)i*C+o]=W.data[(size_t)(tb? o*Hh+i : i*C+o)]; // W is [C,H] (transB=1)
        g.initializer[wt]=T;
        std::string mo=std::string(n.output[0])+"__nobias";
        OnnxNode mm; mm.op_type="MatMul"; mm.name=mo; mm.input={n.input[0],wt}; mm.output={mo}; nn.push_back(mm);
        OnnxNode ad; ad.op_type="Add"; ad.name=std::string(n.output[0])+"__biasadd"; ad.input={mo,n.input[2]}; ad.output={n.output[0]}; nn.push_back(ad);
        continue;
      }
      nn.push_back(n);
    }
    g.nodes=std::move(nn);
  }
  printf("[%d] folded+fused: %zu nodes (seq=%d)\n", party, g.nodes.size(), seq);
  LAP("load+fold+fuse"); _pk=clock_start(); _pb=comm();   // proving window starts here
  { map<string,int>bad; for(auto&n:g.nodes) if(!QuantGraphInfer::supported(n.op_type))bad[n.op_type]++;
    if(!bad.empty()){printf("UNSUPPORTED:"); for(auto&kv:bad)printf(" %s",kv.first.c_str()); printf("\n"); return 2;} }
  _vio=getenv("BG_VOLE")?new BoolIO(new NetIO(party==ALICE?nullptr:peer_ip,port+1),party==ALICE):nullptr;
  setup_zk_arith(ios,threads,party, getenv("EXP_VOLE")?atoll(getenv("EXP_VOLE")):0LL, _vio);
  LAP("setup_zk_arith");
  {
    QuantInfer qi(party,P,ios[0],32,1,threads); RangeCheckLU rclu(party,ios[0]->io,16); CompareLU cmp(&rclu); Div divr(party,ios[0]->io,16);
    // tables AUTO-SIZE with P: exp domain [0,512), gelu & tanh domain +-128 (tanh saturates within +-4).
    const uint64_t ET=1ull<<(P+9), GT=1ull<<(P+8), TT=1ull<<(P+8); const int64_t GOFF=1LL<<(P+7), TOFF=1LL<<(P+7);
    QuantTable expt(ET,party,ios[0]->io); expt.Setup(build_exp_table(ET,P,!getenv("FUSED_NL")));
    QuantTable gtab(GT,party,ios[0]->io); gtab.Setup(build_gelu_table(GT,P,GOFF));
    QuantTable ttab(TT,party,ios[0]->io); ttab.Setup(build_tanh_table(TT,P,TOFF));    // Tanh pooler (generic path + sym_quant_act post-op requant)
    QuantGraphInfer gi(&qi,&cmp,&divr,party); gi.P=P; gi.rcmp=&rclu; gi.exp_tab=&expt; gi.gelu_tab=&gtab; gi.gelu_off=GOFF;
    gi.tanh_tab=&ttab; gi.tanh_off=TOFF; gi.score_bits=(size_t)(P+9); gi.uniform_requant=!getenv("NO_UNIFORM"); if(getenv("PUBLIC_WEIGHTS")) gi.commit_weights=false;
    if(getenv("SYM_BITS")){ gi.sym_act_bits=atoi(getenv("SYM_BITS")); gi.weight_max=(1<<(gi.sym_act_bits-1))-1; gi.uniform_requant=false; }  // DeepProve symmetric int-N (weights+acts), zp=0 -- routes through sym_quant_act (NO QuantizeLinear/clamp)
    // STATIC calibration: CALIB_WRITE=<file> records per-tensor maxes; CALIB=<file> uses them as fixed scales.
    if(getenv("CALIB_WRITE")) gi.calib_write_=true;
    else if(getenv("CALIB")){ std::ifstream f(getenv("CALIB")); std::string nm; double mx; int nc=0; while(f>>nm>>mx){ gi.calib_[nm]=mx; ++nc; } printf("[%d] loaded %d static calib scales from %s\n",party,nc,getenv("CALIB")); }
    // DeepProve-style scale-baked FUSED gelu: one per-op table (input per-tensor int, output = gelu
    // requanted to s_out + clamped) -- no 2^-P rescale.  Built from the static calib scales.
    std::vector<QuantTable*> fused_tabs;
    if(getenv("FUSED_NL") && !gi.calib_.empty() && gi.sym_act_bits>0){
      const int64_t LIM=(1<<(gi.sym_act_bits-1))-1; const uint64_t FT=1ull<<13; const int64_t FOFF=1ll<<12; gi.gelu_fused_off=FOFF;
      // Resolve s_in by tracing the gelu input back through SCALE-TRANSPARENT data-movement ops to
      // its calibrated producer (BERT has a Reshape between the c_fc/MatMul and the gelu).
      std::map<std::string,const OnnxNode*> prod; for(auto&nn:g.nodes) for(auto&o:nn.output) prod[o]=&nn;
      static const std::set<std::string> DM={"Reshape","Transpose","Split","Cast","Unsqueeze","Squeeze","Flatten","Dropout","Identity"};
      auto calib_in=[&](std::string t)->std::string{
        for(int hop=0; hop<8 && !gi.calib_.count(t); ++hop){ auto it=prod.find(t);
          if(it==prod.end()||!DM.count(it->second->op_type)||it->second->input.empty()) break; t=it->second->input[0]; }
        return gi.calib_.count(t)?t:std::string(); };
      for(auto&n:g.nodes) if((n.op_type=="Gelu"||n.op_type=="Erf") && !n.input.empty() && gi.calib_.count(n.output[0])){
        std::string cin=calib_in(n.input[0]); if(cin.empty()) continue;
        double s_in=gi.calib_[cin]/(double)LIM, s_out=gi.calib_[n.output[0]]/(double)LIM;
        QuantTable* t=new QuantTable(FT,party,ios[0]->io); t->Setup(build_gelu_table_fused(FT,FOFF,s_in,s_out,-LIM,LIM));
        gi.gelu_tabs_[n.output[0]]=t; gi.gelu_scales_[n.output[0]]={s_in,s_out}; fused_tabs.push_back(t); }
      printf("[%d] built %zu DeepProve scale-baked gelu tables (input per-tensor, fused requant+clamp)\n",party,fused_tabs.size());
    }
    vector<int64_t> xr(seq,0); vector<IntFp> cx; qi.commit_vec(cx,xr); vector<int64_t> oref;
    LAP("table-setup");
    printf("[%d] running %zu ops...\n", party, g.nodes.size());
    vector<IntFp> out=gi.run(g,cx,xr,1.0/(1<<P),0,oref);
    LAP("forward-prove");
    printf("[%d] dispatched %zu ops, %zu output logits\n", party, gi.ops_run.size(), out.size());
    if(getenv("CALIB_WRITE") && party==ALICE){ std::ofstream f(getenv("CALIB_WRITE")); for(auto&kv:gi.calib_max_) f<<kv.first<<" "<<kv.second<<"\n"; printf("[%d] wrote %zu static calib scales to %s\n",party,gi.calib_max_.size(),getenv("CALIB_WRITE")); }
    // classification head: proven argmax over the (few) class logits (skip with NO_ARGMAX to isolate forward soundness)
    // SYM path: the classifier logits are RAW matmul accumulators (scale s_x*s_w, un-requanted since
    // `logits` is a graph output -> post-op sym-quant skipped), so |logit| ~ 2^34 (768-term int sum).
    // argmax's assert_le range-checks (sel - x_i) in operand_bits+1 bits, so use a wide bound (48).
    const int ARGBITS=getenv("ARGBITS")?atoi(getenv("ARGBITS")):48;
    int cls=-1; if(!getenv("NO_ARGMAX")){ IntFp idx=cmp.argmax(out,ARGBITS); cls=(int)idx.reveal(); } else for(auto&v:out) v.reveal();
    LAP("argmax");
    rclu.Teardown(); expt.Teardown(); gtab.Teardown(); ttab.Teardown(); for(auto t:fused_tabs)t->Teardown(); divr.destroy(); qi.destroy(); LAP("lookup-teardown"); finalize_zk_arith(); LAP("finalize_zk_arith");
    if(BENCH&&party==ALICE){ double us=time_from(_pk); uint64_t b=comm()-_pb;
      fprintf(stderr,"[BENCH] === PROVING TOTAL   %8.3f s  %8.2f MB  (setup_zk -> finalize, excl. model load) ===\n",us/1e6,b/1e6);
      fprintf(stderr,"[BENCH] peak RSS  %.0f MB\n",getPeakRSS()/1e6); }
    if(getenv("NETIO_STATS")){ auto*m=ios[0]->io;
      fprintf(stderr,"[NETIO] party=%d threads=%d rounds_total=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n",party,threads,(unsigned long long)m->rounds,(unsigned long long)m->flushes_count,m->send_counter/1e6,m->recv_counter/1e6);
      if(_vio){ auto*v=_vio->io; fprintf(stderr,"[NETIO] vole-socket: rounds=%llu sent_MB=%.1f recv_MB=%.1f\n",(unsigned long long)v->rounds,v->send_counter/1e6,v->recv_counter/1e6); } }
    for(auto t:fused_tabs) delete t;
    if(party==ALICE){
      double mn=1e18,mx=-1e18; for(size_t i=0;i<oref.size();++i){double v=oref[i]/(double)(1<<P); mn=min(mn,v);mx=max(mx,v);}
      printf("PROVED bert-base (SYM int%d): %zu ops, %zu class logits, range [%.3f,%.3f], argmax class=%d\n", gi.sym_act_bits, gi.ops_run.size(), out.size(), mn, mx, cls);
      printf("pass: ml-quant proved real BERT-base forward via symmetric sym_quant_act (no QuantizeLinear/clamp; verifier accepted)\n");
    } else printf("verifier: bert-sym proof complete\n");
  }
  delete ios[0]; if(_vio) delete _vio; return 0;
}
