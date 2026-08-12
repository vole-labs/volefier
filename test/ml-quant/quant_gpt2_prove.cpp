// Prove a real GPT-2 (gpt2-12_nocache) forward through ml-quant.
// Public: prompt (input_ids), attention_mask(=1), position_ids, per-tensor scales.
// Committed witness: dense-matmul weights (private model, commit_weights=true by default).
// NOTE: embedding table (wte Gather) and LayerNorm gamma/beta are still public model params;
// full model privacy would commit those too -- this covers the dense Gemm/MatMul weights.
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
int port, party; const int threads=getenv("ZK_THREADS")?atoi(getenv("ZK_THREADS")):4; const int P=getenv("P")?atoi(getenv("P")):7;
int main(int argc,char**argv){
  parse_party_and_port(argv,&party,&port); setvbuf(stdout,NULL,_IONBF,0);
  int seq = argc>3 ? atoi(argv[3]) : 6;
  const char* peer_ip = getenv("PEER_IP") ? getenv("PEER_IP") : "127.0.0.1";   // BOB(verifier) connects to ALICE(prover); set PEER_IP=<prover IP> on BOB
  // Single-io engine: ONE main socket carries ALL ZK protocol I/O; `threads` drives the internal
  // compute/VOLE pools, NOT sockets.  BG_VOLE adds exactly one more socket (vole_io on port+1).
  BoolIO* ios[1]; ios[0]=new BoolIO(new NetIO(party==ALICE?nullptr:peer_ip,port),party==ALICE);
  BoolIO* _vio=nullptr;
  // BENCH=1 (or PHASES=1): per-phase wall-time + COMMUNICATION (the VOLE-ZK cost metric), including the
  // LogUp lookup-teardown the [TIMING] op-profiler defers out.  Prover-side (ALICE) figures.
  const bool BENCH=getenv("BENCH")||getenv("PHASES");
  auto comm=[&](){ return io_counter(ios[0]->io) + (_vio ? io_counter(_vio->io) : (uint64_t)0); };
  auto _ck=clock_start(); uint64_t _cb=comm(); auto _pk=_ck; uint64_t _pb=_cb;   // _ck/_cb: per-phase; _pk/_pb: proving window
  uint64_t _cr=0;  // per-phase round counter (main socket direction switches)
  auto LAP=[&](const char*nm){ if(BENCH&&party==ALICE){ double us=time_from(_ck); uint64_t b=comm()-_cb; fprintf(stderr,"[BENCH] %-18s %8.3f s  %8.2f MB  %5llu rounds\n",nm,us/1e6,b/1e6,(unsigned long long)(ios[0]->io->rounds-_cr)); } _ck=clock_start(); _cb=comm(); _cr=ios[0]->io->rounds; };
  const char* mp = getenv("MODEL") ? getenv("MODEL") : "model/gpt2-12_nocache/model.onnx";
  OnnxGraph g=load_onnx(mp);
  auto inj=[&](const char*nm,vector<int64_t>iv){ OnnxTensor t; t.name=nm; t.dtype=7; t.dims={1,(int64_t)iv.size()}; t.i64=iv; g.initializer[nm]=t; g.input_shapes[nm]={1,(int64_t)iv.size()}; };
  vector<int64_t> ids,mask,pos;
  if(getenv("IDS")){string s=getenv("IDS"),c;for(char ch:s+","){if(ch==','){if(!c.empty())ids.push_back(atoll(c.c_str()));c.clear();}else c+=ch;}seq=(int)ids.size();}
  else for(int i=0;i<seq;++i)ids.push_back(100+i);
  for(int i=0;i<seq;++i){mask.push_back(1);pos.push_back(i);}
  inj("input_ids",ids); inj("attention_mask",mask); inj("position_ids",pos);
  FoldStats fs; g=fold_constants(g,fs); FuseStats us; g=fuse_patterns(g,us);
  printf("[%d] folded+fused: %zu nodes (seq=%d)\n", party, g.nodes.size(), seq);
  LAP("load+fold+fuse"); _pk=clock_start(); _pb=comm();   // proving window starts here
  { map<string,int>bad; for(auto&n:g.nodes) if(!QuantGraphInfer::supported(n.op_type))bad[n.op_type]++;
    if(!bad.empty()){printf("UNSUPPORTED:"); for(auto&kv:bad)printf(" %s",kv.first.c_str()); printf("\n"); return 2;} }
  const int64_t _ev=getenv("EXP_VOLE")?atoll(getenv("EXP_VOLE")):0LL;
  _vio=getenv("BG_VOLE")?new BoolIO(new NetIO(party==ALICE?nullptr:peer_ip,port+1),party==ALICE):nullptr;
  const int _vt = getenv("VOLE_THREADS") ? atoi(getenv("VOLE_THREADS")) : -1;   // sVOLE pool size, independent of ostriple `threads`; <0 = same as threads
  setup_zk_arith(ios,threads,party,_ev,_vio,_vt);   // BG_VOLE=1 -> background sVOLE on 2nd socket (streams on demand; EXP_VOLE ignored). EXP_VOLE>0 alone -> prepay
  LAP("setup_zk_arith");
  {
    QuantInfer qi(party,P,ios[0],32,1,threads); RangeCheckLU rclu(party,ios[0]->io,16); CompareLU cmp(&rclu); Div divr(party,ios[0]->io,16);
    // tables AUTO-SIZE with P: exp domain [0,512), gelu domain +-128, mask clamp ~256 -- all P-independent in real units.
    const uint64_t ET=1ull<<(P+9), GT=1ull<<(P+8); const int64_t GOFF=1LL<<(P+7);
    QuantTable expt(ET,party,ios[0]->io); expt.Setup(build_exp_table(ET,P,!getenv("FUSED_NL")));
    QuantTable gtab(GT,party,ios[0]->io); gtab.Setup(build_gelu_table(GT,P,GOFF));
    QuantGraphInfer gi(&qi,&cmp,&divr,party); gi.P=P; gi.rcmp=&rclu; gi.exp_tab=&expt; gi.gelu_tab=&gtab; gi.gelu_off=GOFF; gi.score_bits=(size_t)(P+9); gi.uniform_requant=!getenv("NO_UNIFORM"); if(getenv("PUBLIC_WEIGHTS")) gi.commit_weights=false;
    if(getenv("SYM_BITS")){ gi.sym_act_bits=atoi(getenv("SYM_BITS")); gi.weight_max=(1<<(gi.sym_act_bits-1))-1; gi.uniform_requant=false; }  // DeepProve symmetric int-N (weights+acts), zp=0
    // STATIC calibration: CALIB_WRITE=<file> records per-tensor maxes; CALIB=<file> uses them as fixed scales.
    if(getenv("CALIB_WRITE")) gi.calib_write_=true;
    else if(getenv("CALIB")){ std::ifstream f(getenv("CALIB")); std::string nm; double mx; int nc=0; while(f>>nm>>mx){ gi.calib_[nm]=mx; ++nc; } printf("[%d] loaded %d static calib scales from %s\n",party,nc,getenv("CALIB")); }
    // DeepProve-style scale-baked FUSED nonlinear lookups: one per-op gelu table (input per-tensor int,
    // output = gelu requanted to s_out + clamped) -- no 2^-P rescale.  Built from the static calib scales.
    std::vector<QuantTable*> fused_tabs;
    if(getenv("FUSED_NL") && !gi.calib_.empty() && gi.sym_act_bits>0){
      const int64_t LIM=(1<<(gi.sym_act_bits-1))-1; const uint64_t FT=1ull<<13; const int64_t FOFF=1ll<<12; gi.gelu_fused_off=FOFF;
      // Resolve s_in by tracing the gelu input back through SCALE-TRANSPARENT data-movement ops to its
      // calibrated producer: the folded gpt2 has a Reshape between c_fc/Gemm and the gelu, and (with the
      // data-movement skip) those pass-through tensors aren't in the calib, so keying s_in on the direct
      // input silently disables the fused gelu.  The Reshape doesn't change values/scale, so c_fc/Gemm's
      // calib scale is the correct s_in.
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
    printf("[%d] dispatched %zu ops, %zu logits\n", party, gi.ops_run.size(), out.size());
    if(getenv("CALIB_WRITE") && party==ALICE){ std::ofstream f(getenv("CALIB_WRITE")); for(auto&kv:gi.calib_max_) f<<kv.first<<" "<<kv.second<<"\n"; printf("[%d] wrote %zu static calib scales to %s\n",party,gi.calib_max_.size(),getenv("CALIB_WRITE")); }
    // ---- proven ArgMax over each position's logits -> the predicted next token ----
    // select_max commits a one-hot selector, proves exactly one bit set, and proves the
    // selected logit >= every other logit; argmax() returns the committed winning index.
    const int V=50257; const int npos=(out.size()%V==0)?(int)(out.size()/V):0; vector<int> pred(npos);
    if(npos>0){
      if(getenv("OLD_ARGMAX")){ for(int t=0;t<npos;++t){ vector<IntFp> row(out.begin()+(int64_t)t*V, out.begin()+(int64_t)(t+1)*V);
        IntFp idx=cmp.argmax(row,30); pred[t]=(int)idx.reveal(); } }
      else { vector<IntFp> selv, idxv;   // batched rows: bulk selector commit + RLC bitness, zero mult gates
        cmp.select_max_rows(out,(size_t)npos,(size_t)V,30,selv,idxv);
        for(int t=0;t<npos;++t) pred[t]=(int)idxv[t].reveal(); } }
    else { for(auto&v:out) v.reveal(); }   // partial output (STOP_AFTER bisection): just reveal to trigger finalize
    LAP("argmax");
    rclu.Teardown(); expt.Teardown(); gtab.Teardown(); for(auto t:fused_tabs)t->Teardown(); divr.destroy(); qi.destroy(); LAP("lookup-teardown"); finalize_zk_arith(); LAP("finalize_zk_arith");  // divr/qi destroyed BEFORE finalize: their final range-check batches must be in the proof (else proven post-finalize -> unverified + OOB on the freed exec)
    if(BENCH&&party==ALICE){ double us=time_from(_pk); uint64_t b=comm()-_pb;
      fprintf(stderr,"[BENCH] === PROVING TOTAL   %8.3f s  %8.2f MB  (setup_zk -> finalize, excl. model load) ===\n",us/1e6,b/1e6);
      fprintf(stderr,"[BENCH] peak RSS  %.0f MB\n",getPeakRSS()/1e6); }
    if(getenv("NETIO_STATS")){ auto*m=ios[0]->io;
      fprintf(stderr,"[NETIO] party=%d threads=%d rounds_total=%llu rounds_max=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n",party,threads,(unsigned long long)m->rounds,(unsigned long long)m->rounds,(unsigned long long)m->flushes_count,m->send_counter/1e6,m->recv_counter/1e6);
      if(_vio){ auto*v=_vio->io; fprintf(stderr,"[NETIO] vole-socket: rounds=%llu flushes=%llu sent_MB=%.1f recv_MB=%.1f\n",(unsigned long long)v->rounds,(unsigned long long)v->flushes_count,v->send_counter/1e6,v->recv_counter/1e6); } }    for(auto t:fused_tabs) delete t;
    if(party==ALICE){ printf("predicted next-token ids (proven argmax): "); for(int t=0;t<npos;++t) printf("%d ",pred[t]); printf("\n"); }
    if(party==ALICE){ double mn=1e18,mx=-1e18; for(size_t i=0;i<out.size()&&i<oref.size();++i){double v=oref[i]/(double)(1<<P); mn=min(mn,v);mx=max(mx,v);} 
      printf("PROVED %s: %zu ops, %zu logits, logit range [%.2f,%.2f]\n", mp, gi.ops_run.size(), out.size(), mn, mx);
      printf("pass: ml-quant proved GPT-2 forward [%s] (verifier accepted)\n", mp); }
    else printf("verifier: GPT-2 proof complete [%s]\n", mp);
  }
  delete ios[0]; if(_vio) delete _vio; return 0;
}
