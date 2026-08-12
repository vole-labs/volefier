// Self-contained test of the SafeTensors reader: hand-construct a tiny
// .safetensors file (F32 + BF16 tensors), read it back, verify shapes/values.
//   test_io_safetensors

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include "volefier/io/safetensors.h"

using namespace volefier;
using namespace std;

static void put_u64(string &s, uint64_t v) { for (int i = 0; i < 8; ++i) s.push_back((char)((v >> (8 * i)) & 0xff)); }
static void put_f32(string &s, float v) { uint32_t b; memcpy(&b, &v, 4); for (int i = 0; i < 4; ++i) s.push_back((char)((b >> (8 * i)) & 0xff)); }
static void put_bf16(string &s, float v) { uint32_t b; memcpy(&b, &v, 4); uint16_t h = (uint16_t)(b >> 16); s.push_back((char)(h & 0xff)); s.push_back((char)(h >> 8)); }

int main() {
  // data region: w (F32 [2,2], 16 B) at [0,16); b (F32 [2], 8 B) at [16,24); c (BF16 [3], 6 B) at [24,30)
  string data;
  put_f32(data, 1.0f); put_f32(data, 2.0f); put_f32(data, 3.0f); put_f32(data, 4.0f);   // w
  put_f32(data, 0.5f); put_f32(data, -0.5f);                                             // b
  put_bf16(data, 1.0f); put_bf16(data, 2.0f); put_bf16(data, 0.25f);                     // c (exact in bf16)

  string header =
      "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
      "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]},"
      "\"c\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[24,30]},"
      "\"__metadata__\":{\"format\":\"pt\"}}";

  string file; put_u64(file, header.size()); file += header; file += data;
  { ofstream o("/tmp/test_model.safetensors", ios::binary); o.write(file.data(), file.size()); }

  SafeTensors st("/tmp/test_model.safetensors");
  int fails = 0;
  auto check = [&](bool ok, const string &msg) { if (!ok) { ++fails; cout << "  FAIL: " << msg << endl; } };

  check(st.has("w") && st.has("b") && st.has("c"), "all three tensors present");
  const auto &w = st.at("w"), &b = st.at("b"), &c = st.at("c");
  check(w.shape == vector<int64_t>({2, 2}), "w shape [2,2]");
  check(w.data == vector<float>({1, 2, 3, 4}), "w values");
  check(b.shape == vector<int64_t>({2}) && b.data == vector<float>({0.5f, -0.5f}), "b values");
  check(c.shape == vector<int64_t>({3}) && c.data == vector<float>({1.0f, 2.0f, 0.25f}), "c (bf16) values");

  if (fails) { cout << fails << " checks failed" << endl; return 1; }
  cout << "pass SafeTensors reader (F32 + BF16, " << st.tensors.size() << " tensors)" << endl;
  return 0;
}
