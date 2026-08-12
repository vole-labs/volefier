// PER-TENSOR quantized transformer block (zkGPT/DeepProve method): every matmul
// activation (Q,K,V, attention scores, context, FFN hidden) carries its OWN
// scale, and each matmul requants by S_in*S_w/S_out -> a distinct c*2^e. softmax
// and GeLU use scale-aware tables (built for their per-tensor input scale);
// outputs are anchored at 2^-P. LayerNorm is omitted (it is scale-agnostic --
// the z-score normalizes the input scale out -- and is validated separately).
//
//   test_ml-quant_transformer_pertensor 1 12345
//   test_ml-quant_transformer_pertensor 2 12345

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-quant/quant_transformer.h"
#include "test/bench_e2e.h"

using namespace emp;
using namespace std;

int port, party;
const int threads = 4;
const int P = 12;

static int64_t rcm(int64_t acc, int64_t c, int M) { int64_t N = acc*c, h = (int64_t)1<<(M-1), D = (int64_t)1<<M; return N>=0?(N+h)/D:-(((-N)+h)/D); }
static int64_t rdiv(int64_t N, int64_t D) { return N>=0?(N+D/2)/D:-(((-N)+D/2)/D); }
static PRG *gprg;
static int64_t rnd(int64_t lo, int64_t hi) { if (party != ALICE) return 0; uint64_t r; gprg->random_data(&r, sizeof(r)); return lo + (int64_t)(r % (uint64_t)(hi - lo + 1)); }

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "----- per-tensor quantized transformer block (zkGPT/DeepProve) -----" << endl << endl;

  PRG prg; gprg = &prg;
  BenchE2E<BoolIO> _bench(party, ios, threads);
  setup_zk_arith(ios, threads, party);
  QuantInfer qi(party, P, ios[0], /*out_bits=*/20, /*wc=*/1, /*mat_threads=*/threads);
  RangeCheckLU rclu(party, ios[0]->io, 16);
  CompareLU cmp(&rclu);
  Div divr(party, ios[0]->io, /*range_bits=*/24);

  const int s = 4, d = 8, h = 2, dh = d / h, ffn = 16;
  const double Sres = 1.0/256, Sw = 1.0/200, SQ = 1.0/300, SK = 1.0/280, SV = 1.0/320,
               Ssc = 1.0/256, Sctx = 1.0/290, Sh = 1.0/310, Sa = 1.0/(1<<P);   // Sa = 2^-P anchor
  // per-tensor requant multipliers
  auto srch = [](double r){ auto ce = QuantInfer::scale_search(r); return pair<int64_t,int>{(int64_t)ce.first, (int)(-ce.second)}; };
  auto MQ = srch(Sres*Sw/SQ), MK = srch(Sres*Sw/SK), MV = srch(Sres*Sw/SV);
  auto MQK = srch(SQ*SK/(sqrt((double)dh)*Ssc));        // Q.K^T * 1/sqrt(dh) -> S_score
  auto MAV = srch(Sa*SV/Sctx);                          // probs(2^-P).V -> S_ctx
  auto MO = srch(Sctx*Sw/Sres), M1 = srch(Sres*Sw/Sh), M2 = srch(Sa*Sw/Sres);

  // scale-aware tables: exp for S_score, gelu for S_h
  const uint64_t ET = 1u<<14, GT = 1u<<14; const int64_t GOFF = 1<<13;
  QuantTable expt(ET, party, ios[0]->io); expt.Setup(build_exp_table_s(ET, P, Ssc));
  QuantTable gtab(GT, party, ios[0]->io); gtab.Setup(build_gelu_table_s(GT, P, GOFF, Sh));
  vector<uint64_t> exp_ref = build_exp_table_s(ET, P, Ssc), gelu_ref = build_gelu_table_s(GT, P, GOFF, Sh);
  QuantSoftmax softmax(&qi, &cmp, &expt, &divr, P, 16);
  QuantActivation gelu(&gtab, GOFF);

  // committed inputs (small integers at S_res)
  auto rmat = [&](int r, int c, int64_t lo, int64_t hi){ vector<vector<int64_t>> m(r, vector<int64_t>(c)); for(auto&row:m)for(auto&v:row)v=rnd(lo,hi); return m; };
  auto rvec = [&](int n, int64_t lo, int64_t hi){ vector<int64_t> v(n); for(auto&e:v)e=rnd(lo,hi); return v; };
  auto H = rmat(s, d, -100, 100);
  auto Wq = rmat(d,d,-3,3), Wk = rmat(d,d,-3,3), Wv = rmat(d,d,-3,3), Wo = rmat(d,d,-3,3);
  auto W1 = rmat(ffn,d,-3,3), W2 = rmat(d,ffn,-3,3);
  auto bq=rvec(d,-20,20),bk=rvec(d,-20,20),bv=rvec(d,-20,20),bo=rvec(d,-20,20),b1=rvec(ffn,-20,20),b2=rvec(d,-20,20);

  Mat cH; qi.commit_mat(cH, H);
  Mat cWq,cWk,cWv,cWo,cW1,cW2; qi.commit_mat(cWq,Wq);qi.commit_mat(cWk,Wk);qi.commit_mat(cWv,Wv);qi.commit_mat(cWo,Wo);qi.commit_mat(cW1,W1);qi.commit_mat(cW2,W2);
  vector<IntFp> cbq,cbk,cbv,cbo,cb1,cb2; qi.commit_vec(cbq,bq);qi.commit_vec(cbk,bk);qi.commit_vec(cbv,bv);qi.commit_vec(cbo,bo);qi.commit_vec(cb1,b1);qi.commit_vec(cb2,b2);

  // ----- committed forward -----
  Mat Q(s),K(s),V(s);
  for (int i=0;i<s;i++){ qi.dense_layer_q(Q[i],cWq,cbq,cH[i],Sres,Sw,SQ); qi.dense_layer_q(K[i],cWk,cbk,cH[i],Sres,Sw,SK); qi.dense_layer_q(V[i],cWv,cbv,cH[i],Sres,Sw,SV); }
  Mat ctx(s, vector<IntFp>(d));
  for (int i=0;i<s;i++) for (int hh=0;hh<h;hh++){ int off=hh*dh;
    vector<IntFp> qh(Q[i].begin()+off, Q[i].begin()+off+dh), scaled(i+1);
    for (int j=0;j<=i;j++){ vector<IntFp> kh(K[j].begin()+off,K[j].begin()+off+dh); scaled[j]=qi.requant_cm_one(qi.dot_ip(qh,kh),(uint64_t)MQK.first,(size_t)MQK.second); }
    vector<IntFp> probs = softmax.forward(scaled);
    for (int k=0;k<dh;k++){ vector<IntFp> vk(i+1); for(int j=0;j<=i;j++) vk[j]=V[j][off+k]; ctx[i][off+k]=qi.requant_cm_one(qi.dot_ip(probs,vk),(uint64_t)MAV.first,(size_t)MAV.second); }
  }
  Mat attn(s);
  for (int i=0;i<s;i++) qi.dense_layer_q(attn[i],cWo,cbo,ctx[i],Sctx,Sw,Sres);
  Mat a(s); for (int i=0;i<s;i++){ a[i].resize(d); for(int j=0;j<d;j++) a[i][j]=cH[i][j]+attn[i][j]; }
  Mat out(s);
  for (int i=0;i<s;i++){ vector<IntFp> hid, go, ff; qi.dense_layer_q(hid,cW1,cb1,a[i],Sres,Sw,Sh); gelu.forward(go,hid); qi.dense_layer_q(ff,cW2,cb2,go,Sa,Sw,Sres); out[i].resize(d); for(int j=0;j<d;j++) out[i][j]=a[i][j]+ff[j]; }

  // ----- scale-aware integer reference (ALICE) -----
  size_t fails = 0;
  if (party == ALICE) {
    auto dq = [&](const vector<vector<int64_t>>&W, const vector<int64_t>&b, const vector<int64_t>&x, pair<int64_t,int> m){ vector<int64_t> y(W.size()); for(size_t j=0;j<W.size();j++){ int64_t acc=0; for(size_t i=0;i<x.size();i++) acc+=W[j][i]*x[i]; y[j]=rcm(acc,m.first,m.second)+b[j]; } return y; };
    vector<vector<int64_t>> Qr(s),Kr(s),Vr(s),ctxr(s, vector<int64_t>(d)),attnr(s),ar(s),outr(s);
    for(int i=0;i<s;i++){ Qr[i]=dq(Wq,bq,H[i],MQ); Kr[i]=dq(Wk,bk,H[i],MK); Vr[i]=dq(Wv,bv,H[i],MV); }
    for(int i=0;i<s;i++) for(int hh=0;hh<h;hh++){ int off=hh*dh; vector<int64_t> sc(i+1);
      for(int j=0;j<=i;j++){ int64_t a2=0; for(int k=0;k<dh;k++) a2+=Qr[i][off+k]*Kr[j][off+k]; sc[j]=rcm(a2,MQK.first,MQK.second); }
      int64_t mx=sc[0]; for(auto v:sc) mx=max(mx,v); int64_t sum=0; vector<int64_t> e(i+1);
      for(int j=0;j<=i;j++){ e[j]=(int64_t)exp_ref[mx-sc[j]]; sum+=e[j]; }
      vector<int64_t> pr(i+1); for(int j=0;j<=i;j++) pr[j]=rdiv(e[j]*(1<<P),sum);
      for(int k=0;k<dh;k++){ int64_t a2=0; for(int j=0;j<=i;j++) a2+=pr[j]*Vr[j][off+k]; ctxr[i][off+k]=rcm(a2,MAV.first,MAV.second); }
    }
    for(int i=0;i<s;i++) attnr[i]=dq(Wo,bo,ctxr[i],MO);
    for(int i=0;i<s;i++){ ar[i].resize(d); for(int j=0;j<d;j++) ar[i][j]=H[i][j]+attnr[i][j]; }
    for(int i=0;i<s;i++){ auto hid=dq(W1,b1,ar[i],M1); vector<int64_t> go(ffn); for(int j=0;j<ffn;j++) go[j]=QuantInfer::decode_signed(gelu_ref[hid[j]+GOFF]); auto ff=dq(W2,b2,go,M2); outr[i].resize(d); for(int j=0;j<d;j++) outr[i][j]=ar[i][j]+ff[j]; }
    for(int i=0;i<s;i++) for(int j=0;j<d;j++){ int64_t got=QuantInfer::decode_signed(out[i][j].reveal()); if(got!=outr[i][j]){ if(fails<5) cout<<"out["<<i<<"]["<<j<<"] "<<got<<" vs "<<outr[i][j]<<endl; fails++; } }
  } else {
    for(int i=0;i<s;i++) for(int j=0;j<d;j++) out[i][j].reveal();
  }

  rclu.Teardown(); expt.Teardown(); gtab.Teardown(); divr.destroy(); qi.destroy();
  finalize_zk_arith();
  _bench.report("transformer-pertensor (ml-quant)");
  if (party == ALICE) {
    if (fails) { cout << fails << " mismatches" << endl; error("per-tensor transformer failed\n"); }
    cout << "multipliers (c,shift): Q=(" << MQ.first << "," << MQ.second << ") QK=(" << MQK.first << "," << MQK.second
         << ") AV=(" << MAV.first << "," << MAV.second << ") O=(" << MO.first << "," << MO.second << ")" << endl;
    cout << "pass per-tensor quantized transformer block (Q/K/V/score/ctx/hidden each own scale, c!=1)" << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
