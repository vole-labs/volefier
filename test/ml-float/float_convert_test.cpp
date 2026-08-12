// Validate the ported Int62ToFloat / FloatToInt62 conversion circuits (new emp API).
// Round-trip: int -> float -> int, and check the float value matches iv * 2^-s.
#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "volefier/compat.h"
#include "volefier/ml-float/float_convert.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace emp;
using namespace volefier;

int main(int argc, char **argv) {
  int party, port;
  parse_party_and_port(argv, &party, &port);
  BoolIO *io = new BoolIO(new NetIO(party == ALICE ? nullptr : "127.0.0.1", port), party == ALICE);
  ZKBoolSession sess(io, party);
  auto &ctx = sess.ctx();

  const int s = 12; // fixed-point scale 2^-12
  std::vector<int64_t> vals = {0, 1 << 12, -(1 << 12), 3 << 12, 12345, -12345,
                               (int64_t)1 << 20, -((int64_t)1 << 20), 100 << 12, -(100 << 12)};
  const uint64_t PRv = (1ULL << 61) - 1; // field prime, Mersenne
  auto to_field = [&](int64_t v) { return v >= 0 ? (uint64_t)v : PRv - (uint64_t)(-v); };
  auto from_field = [&](uint64_t v) { return v > PRv / 2 ? (int64_t)v - (int64_t)PRv : (int64_t)v; };
  bool ok = true;
  for (int64_t iv : vals) {
    ZKInt x = sess.input<ZKInt>(ALICE, (int64_t)to_field(iv), 62); // field-rep in
    FloatZK f = Int62ToFloat(ctx, x, s);
    auto fv = sess.reveal(f, PUBLIC);       // optional<float>
    ZKInt back = FloatToInt62(ctx, f, s);
    auto iv2 = sess.reveal(back, PUBLIC);   // optional<int64_t> (field-rep)
    if (party == ALICE) {
      double expect = (double)iv / (double)(1 << s);
      double got = (double)fv.value_or(0.0f);
      long long rt = (long long)from_field((uint64_t)iv2.value_or(0));
      bool fok = std::fabs(got - expect) <= 1e-3 * std::max(1.0, std::fabs(expect));
      bool iok = (rt == iv);
      printf("iv=%-9lld  float=%-12.6f (expect %-12.6f) %s   roundtrip int=%-9lld %s\n",
             (long long)iv, got, expect, fok ? "OK" : "BAD", rt, iok ? "OK" : "BAD");
      ok = ok && fok && iok;
    }
  }
  sess.finalize();
  if (party == ALICE) printf(ok ? "pass float_convert\n" : "FAIL float_convert\n");
  delete io;
  return 0;
}
