#ifndef ML_FLOAT_FLOAT_CONVERT_H
#define ML_FLOAT_FLOAT_CONVERT_H

// IEEE-754 <-> fixed-point conversion circuits for the ml-float (Mystique) backend,
// ported to the new emp session/typed-wire API from the reference implementation in
// references/emp-zk/emp-zk/extensions/floats.h (old emp `Integer`/`Float`).  These are
// the two primitives the new emp dropped:
//   FloatToInt62(f, s) : IEEE fp32 f  ->  signed 62-bit two's-complement at scale 2^s
//   Int62ToFloat(x, s) : signed 62-bit two's-complement at scale 2^s  ->  IEEE fp32
// Everything else the float ResNet needs (mul/add/sqrt/rsqrt/less_than/select, arith<->bool)
// already exists in the new emp; these close the last gap.
//
// fp32 wire layout (Float_T<Ctx,32>::w):  w[0..22] mantissa, w[23..30] exponent, w[31] sign.

#include "emp-zk/emp-zk.h"
#include "emp-tool/emp-tool.h"

namespace volefier {

using Ctx    = emp::ZKBoolContext;
using Wire   = Ctx::Wire;
using FloatZK = emp::Float_T<Ctx, 32>;
using Bit    = emp::Bit_T<Ctx>;
using emp::ZKInt;
template <int N> using U = emp::UInt_T<Ctx, N>;

// --- FloatToInt62: fp32 -> signed 62-bit fixed-point at scale 2^s -----------------
inline ZKInt FloatToInt62(Ctx &ctx, const FloatZK &in, int s) {
  const Wire z = ctx.public_bit(false);

  // 24-bit mantissa (23 stored + implicit leading 1) placed in a 61-bit register.
  U<61> frac(ctx);
  for (int i = 0; i < 23; ++i) frac.w[i] = in.w[i];
  frac.w[23] = ctx.public_bit(true);
  for (int i = 24; i < 61; ++i) frac.w[i] = z;

  // exponent (8 bits) minus bias (127 + 23 - s); two's-complement 8-bit.
  U<8> exp(ctx);
  for (int i = 0; i < 8; ++i) exp.w[i] = in.w[23 + i];
  U<8> e    = exp - U<8>::constant(ctx, (uint64_t)(127 + 23 - s));
  U<8> nege = U<8>::constant(ctx, 0) - e;
  Bit  neg  = e[7]; // e < 0  => shift RIGHT by -e, else shift LEFT by e

  // zero-extend the shift magnitudes to the value width for the barrel shifter.
  U<61> shl(ctx), shr(ctx);
  for (int i = 0; i < 8; ++i)  { shl.w[i] = e.w[i];  shr.w[i] = nege.w[i]; }
  for (int i = 8; i < 61; ++i) { shl.w[i] = z;       shr.w[i] = z; }
  U<61> mag = (frac << shl).select(neg, frac >> shr); // neg ? (frac>>shr) : (frac<<shl)

  // two's-complement sign into 62 bits.
  U<62> mag62(ctx);
  for (int i = 0; i < 61; ++i) mag62.w[i] = mag.w[i];
  mag62.w[61] = z;
  U<62> out = mag62.select(in[31], U<62>::constant(ctx, 0) - mag62); // sign ? -mag : mag
  // signed two's-complement -> field representation [0, PR), PR = 2^61 - 1 (Mersenne).
  const uint64_t PRv = (1ULL << 61) - 1;
  out = out.select(out[61], out + U<62>::constant(ctx, PRv)); // out<0 ? out+PR : out
  return ZKInt::from_wires(ctx, out.data(), 62);
}

// --- Int62ToFloat: signed 62-bit fixed-point at scale 2^s -> fp32 -----------------
inline FloatZK Int62ToFloat(Ctx &ctx, const ZKInt &in, int s) {
  // load the 62-bit value in FIELD representation [0, PR), PR = 2^61 - 1 (Mersenne),
  // and map it to signed two's-complement: values > PR/2 represent negatives.
  U<62> xf(ctx);
  for (int i = 0; i < 62; ++i) xf.w[i] = in.data()[i];
  const uint64_t PRv = (1ULL << 61) - 1;
  Bit  fneg = xf > U<62>::constant(ctx, (1ULL << 60) - 1);
  U<62> x = xf.select(fneg, xf - U<62>::constant(ctx, PRv)); // field>PR/2 ? field-PR : field
  Bit  sgn = x[61];
  U<62> mag62 = x.select(sgn, U<62>::constant(ctx, 0) - x); // |x|

  // drop to 61-bit magnitude, find the top set bit.
  U<61> m(ctx);
  for (int i = 0; i < 61; ++i) m.w[i] = mag62.w[i];
  auto lz = m.leading_zeros();                 // UInt<Ctx, bits_for(61)>
  const int lzw = (int)lz.w.size();
  U<8> lz8(ctx);
  for (int i = 0; i < 8; ++i) lz8.w[i] = (i < lzw) ? lz.w[i] : ctx.public_bit(false);
  U<8> firstOne = U<8>::constant(ctx, 60) - lz8; // index of MSB set bit

  // align the mantissa so the leading 1 lands at bit 23.
  U<8> tw = U<8>::constant(ctx, 23);
  Bit  left = firstOne >= tw;
  // A.select(sel,B) == sel ? B : A, so put the false-branch first.
  U<8> off  = (tw - firstOne).select(left, firstOne - tw); // left ? firstOne-23 : 23-firstOne
  U<61> off61(ctx);
  for (int i = 0; i < 8; ++i)  off61.w[i] = off.w[i];
  for (int i = 8; i < 61; ++i) off61.w[i] = ctx.public_bit(false);
  U<61> shifted = (m << off61).select(left, m >> off61); // left ? m>>off : m<<off

  // biased exponent = firstOne + (127 - s).
  U<8> exponent = firstOne + U<8>::constant(ctx, (uint64_t)(127 - s));

  // assemble fp32 wires: [0..22]=mantissa, [23..30]=exponent, [31]=sign.
  FloatZK out = FloatZK::constant(ctx, 0.0f);
  for (int i = 0; i < 23; ++i) out.w[i] = shifted.w[i];
  for (int i = 0; i < 8;  ++i) out.w[23 + i] = exponent.w[i];
  out.w[31] = sgn.w;
  return out;
}

} // namespace volefier

#endif // ML_FLOAT_FLOAT_CONVERT_H
