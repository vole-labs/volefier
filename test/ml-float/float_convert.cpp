// Prototype: the arith<->float round-trip at one boundary, using emp-zk's
// edaBit (arith2bool/bool2arith) + Mystique Fix<->Float (Int62ToFloat /
// FloatToInt62). Commits fixed-point IntFp values, converts to IEEE-754 Float,
// applies ReLU in the Boolean proof, converts back, and checks both the float
// reveal and the round-tripped fixed-point value.
//
//   test_ml-float_float_convert 1 12345
//   test_ml-float_float_convert 2 12345

#include <iostream>
#include <vector>
#include <cmath>
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/ml-float/float_nonlinear.h"

using namespace emp;
using namespace volefier;
using namespace std;

int port, party;
const int threads = 1;
const int S = 16;                       // fixed-point precision (Mystique s=16)
typedef BoolIO<NetIO> ZIO;

static IntFp commit_fx(double real) {    // encode real at scale 2^S into the field
  int64_t fx = llround(real * (double)(1LL << S));
  uint64_t v = fx >= 0 ? (uint64_t)fx : PR - (uint64_t)(-fx);
  return IntFp(v, ALICE);
}
static int64_t dec(uint64_t v) { return v > PR / 2 ? (int64_t)v - (int64_t)PR : (int64_t)v; }

int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  ZIO *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new ZIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i), party == ALICE);
  cout << endl << "----- arith<->float conversion round-trip (Mystique edaBit + Fix<->Float) -----" << endl << endl;

  setup_zk_bool<ZIO>(ios, threads, party);                 // Boolean ZK first
  setup_zk_arith<ZIO>(ios, threads, party, /*enable_conversion=*/true);  // shares VOLE + edaBit

  int fails = 0;
  for (double real : {2.5, -3.25, 0.0, 7.125, -0.5}) {
    IntFp x = commit_fx(real);                  // fixed-point, scale 2^S
    Integer xb = arith2bool<ZIO>(x);            // -> 62-bit Integer (A2B)
    Float xf = Int62ToFloat(xb, S);             // -> IEEE-754 Float (= real)
    Float rf = FloatNonlinear::relu(xf);        // ReLU in float
    double gf = rf.reveal<double>(PUBLIC);

    Integer rb = FloatToInt62(rf, S);           // Float -> fixed-point
    IntFp ri = bool2arith<ZIO>(rb);             // B2A -> IntFp
    int64_t gv = dec(ri.reveal());

    if (party == ALICE) {
      double want = real > 0 ? real : 0.0;
      int64_t want_fx = llround(want * (double)(1LL << S));
      if (fabs(gf - want) > 1e-4 || gv != want_fx) {
        ++fails;
        cout << "  MISMATCH real=" << real << ": float " << gf << " (want " << want
             << "), fixed " << gv << " (want " << want_fx << ")" << endl;
      }
    }
  }

  finalize_zk_arith<ZIO>();
  bool cheated = finalize_zk_bool<ZIO>();
  if (party == ALICE) {
    if (fails) error("conversion round-trip failed\n");
    cout << "pass arith<->float round-trip (A2B + Int62ToFloat + ReLU + FloatToInt62 + B2A)" << endl;
  } else {
    cout << "verifier check: " << (cheated ? "REJECT" : "OK") << endl;
  }
  for (int i = 0; i < threads; ++i) delete ios[i];
  return 0;
}
