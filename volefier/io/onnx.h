#ifndef ML_IO_ONNX_H
#define ML_IO_ONNX_H

// Minimal, dependency-free ONNX reader. ONNX files are protobuf-encoded
// ModelProto messages; this decodes just enough of the wire format to recover
// the operator graph (nodes with op_type / inputs / outputs / attributes) and
// the weight initializers. It is format-level only: it does not interpret the
// operators — that is the job of the graph prover, which dispatches each node to
// a ZKP component and aborts on any operator it does not support.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace volefier {

struct OnnxTensor {
  std::string name;
  std::vector<int64_t> dims;
  std::vector<float> data;
  std::vector<int64_t> i64;
  int dtype = 1;
  bool ext = false; // weight stored in an external .data file (large LLMs)
  std::string ext_loc;
  long long ext_off = 0, ext_len = 0;
  std::string ext_path; // LAZY_INIT: resolved absolute path, decode on demand (data left empty)
};
// LAZY_INIT: decode a still-external tensor from its .data file NOW (no-op if already resolved).
// Lets the graph runner load each weight only when its op runs and free it right after (FREE_INIT),
// so resident fp32 is O(one weight) instead of the whole model.
inline void decode_raw(OnnxTensor &t, const std::string &raw);
inline void resolve_ext(OnnxTensor &t) {
  if (!t.ext || t.ext_path.empty() || !t.data.empty() || !t.i64.empty()) return;
  std::ifstream ef(t.ext_path, std::ios::binary);
  if (!ef) throw std::runtime_error("onnx: cannot reopen external data " + t.ext_path);
  ef.seekg(0, std::ios::end); long long fsz = (long long)ef.tellg();
  long long len = t.ext_len > 0 ? t.ext_len : fsz - t.ext_off;
  std::string slice((size_t)len, '\0');
  ef.seekg((std::streamoff)t.ext_off); ef.read(&slice[0], (std::streamsize)len);
  decode_raw(t, slice);
}
// Read a single fp32 element (index i) of a tensor without decoding the whole thing -- for cheap
// sampling (tie-detection) of a lazy (still-external) weight; falls back to data[] if resolved.
inline float peek_ext_f32(const OnnxTensor &t, long long i) {
  if (!t.data.empty()) return t.data[(size_t)i];
  std::ifstream ef(t.ext_path, std::ios::binary);
  float v = 0; ef.seekg((std::streamoff)(t.ext_off + i * 4)); ef.read((char *)&v, 4); return v;
}
struct OnnxAttr {
  std::string name;
  float f = 0;
  int64_t i = 0;
  std::vector<int64_t> ints;
  OnnxTensor t;
  bool has_t = false;
};
struct OnnxNode {
  std::string op_type, name;
  std::vector<std::string> input, output;
  std::map<std::string, OnnxAttr> attr;
};
struct OnnxGraph {
  std::vector<OnnxNode> nodes;                              // in file (topological) order
  std::map<std::string, OnnxTensor> initializer;            // weights, by name
  std::vector<std::string> inputs, outputs;                 // graph I/O names
  std::map<std::string, std::vector<int64_t>> input_shapes; // declared input dims
};

// ---- minimal protobuf wire reader ----
class PB {
public:
  PB(const char *p, const char *e) : p_(p), e_(e) {}
  bool done() const {
    return p_ >= e_;
  }
  uint64_t varint() {
    uint64_t v = 0;
    int s = 0;
    while (p_ < e_) {
      uint8_t b = (uint8_t)*p_++;
      v |= (uint64_t)(b & 0x7f) << s;
      if (!(b & 0x80)) break;
      s += 7;
    }
    return v;
  }
  void tag(int &field, int &wire) {
    uint64_t t = varint();
    field = (int)(t >> 3);
    wire = (int)(t & 7);
  }
  std::string bytes() {
    uint64_t n = varint();
    std::string s(p_, p_ + n);
    p_ += n;
    return s;
  }
  float f32() {
    float v;
    std::memcpy(&v, p_, 4);
    p_ += 4;
    return v;
  }
  void skip(int wire) {
    if (wire == 0)
      varint();
    else if (wire == 2) {
      uint64_t n = varint();
      p_ += n;
    } else if (wire == 1)
      p_ += 8;
    else if (wire == 5)
      p_ += 4;
  }

private:
  const char *p_, *e_;
};

inline OnnxTensor parse_tensor(const std::string &b); // fwd decl (Constant's value attr)

inline OnnxAttr parse_attr(const std::string &b) {
  OnnxAttr a;
  PB r(b.data(), b.data() + b.size());
  int f, w;
  while (!r.done()) {
    r.tag(f, w);
    if (f == 1 && w == 2)
      a.name = r.bytes(); // name
    else if (f == 2 && w == 5)
      a.f = r.f32(); // f (float)
    else if (f == 3 && w == 0)
      a.i = (int64_t)r.varint(); // i (int)
    else if (f == 5 && w == 2) {
      a.t = parse_tensor(r.bytes());
      a.has_t = true;
    } // t (TensorProto, e.g. Constant value)
    else if (f == 8 && w == 2) {
      std::string d = r.bytes();
      PB rr(d.data(), d.data() + d.size());
      while (!rr.done()) a.ints.push_back((int64_t)rr.varint());
    } // ints (packed)
    else if (f == 8 && w == 0)
      a.ints.push_back((int64_t)r.varint()); // ints (unpacked)
    else
      r.skip(w);
  }
  return a;
}

// decode a raw byte blob (inline raw_data or external-data file slice) into t per dtype.
inline void decode_raw(OnnxTensor &t, const std::string &raw) {
  if (raw.empty()) return;
  if (t.dtype == 7) {
    int n = (int)raw.size() / 8; t.i64.resize(n); std::memcpy(t.i64.data(), raw.data(), raw.size());
  } else if (t.dtype == 3) {
    t.i64.resize(raw.size()); for (size_t i = 0; i < raw.size(); ++i) t.i64[i] = (int64_t)(int8_t)raw[i];
  } else if (t.dtype == 2) {
    t.i64.resize(raw.size()); for (size_t i = 0; i < raw.size(); ++i) t.i64[i] = (int64_t)(uint8_t)raw[i];
  } else if (t.dtype == 6) {
    int n = (int)raw.size() / 4; t.i64.resize(n);
    for (int i = 0; i < n; ++i) { int32_t x; std::memcpy(&x, raw.data() + 4 * i, 4); t.i64[i] = x; }
  } else if (t.dtype == 9) {
    t.i64.resize(raw.size()); for (size_t i = 0; i < raw.size(); ++i) t.i64[i] = (int64_t)(uint8_t)raw[i];
  } else if (t.data.empty()) {
    size_t n = raw.size() / 4; t.data.resize(n); if (n) std::memcpy(t.data.data(), raw.data(), n * 4);
  }
}

inline OnnxTensor parse_tensor(const std::string &b) {
  OnnxTensor t;
  std::string raw;
  PB r(b.data(), b.data() + b.size());
  int f, w;
  while (!r.done()) {
    r.tag(f, w);
    if (f == 14 && w == 0) {
      if (r.varint() == 1) t.ext = true; // data_location = EXTERNAL (field 14, not 12=doc_string)
    } else if (f == 13 && w == 2) {       // external_data: StringStringEntryProto (key,value)
      t.ext = true;                       // presence of external_data => external tensor
      std::string d = r.bytes();
      PB rr(d.data(), d.data() + d.size());
      std::string key, val; int ff, ww;
      while (!rr.done()) { rr.tag(ff, ww); if (ff == 1 && ww == 2) key = rr.bytes(); else if (ff == 2 && ww == 2) val = rr.bytes(); else rr.skip(ww); }
      if (key == "location") t.ext_loc = val;
      else if (key == "offset") t.ext_off = atoll(val.c_str());
      else if (key == "length") t.ext_len = atoll(val.c_str());
    } else if (f == 1 && w == 0)
      t.dims.push_back((int64_t)r.varint()); // dims (unpacked)
    else if (f == 1 && w == 2) {
      std::string d = r.bytes();
      PB rr(d.data(), d.data() + d.size());
      while (!rr.done()) t.dims.push_back((int64_t)rr.varint());
    } // dims (packed)
    else if (f == 2 && w == 0)
      t.dtype = (int)r.varint(); // data_type (1=float,7=int64)
    else if (f == 8 && w == 2)
      t.name = r.bytes(); // name
    else if (f == 9 && w == 2)
      raw = r.bytes(); // raw_data
    else if (f == 4 && w == 2) {
      std::string d = r.bytes();
      size_t n = d.size() / 4;
      t.data.resize(n);
      if (n) std::memcpy(t.data.data(), d.data(), n * 4);
    } // float_data (packed, guarded)
    else if (f == 5 && w == 2) {
      std::string d = r.bytes();
      PB rr(d.data(), d.data() + d.size());
      while (!rr.done()) t.i64.push_back((int32_t)(uint32_t)rr.varint());
    } // int32_data (packed, e.g. zero_points)
    else if (f == 5 && w == 0)
      t.i64.push_back((int32_t)(uint32_t)r.varint()); // int32_data (unpacked)
    else if (f == 7 && w == 2) {
      std::string d = r.bytes();
      int n = (int)d.size() / 8;
      t.i64.resize(n);
      std::memcpy(t.i64.data(), d.data(), d.size());
    } // int64_data (packed)
    else
      r.skip(w);
  }
  decode_raw(t, raw); // inline raw_data; external-data tensors are resolved later in load_onnx
  return t;
}

inline OnnxNode parse_node(const std::string &b) {
  OnnxNode n;
  PB r(b.data(), b.data() + b.size());
  int f, w;
  while (!r.done()) {
    r.tag(f, w);
    if (f == 1 && w == 2)
      n.input.push_back(r.bytes()); // input
    else if (f == 2 && w == 2)
      n.output.push_back(r.bytes()); // output
    else if (f == 3 && w == 2)
      n.name = r.bytes(); // name
    else if (f == 4 && w == 2)
      n.op_type = r.bytes(); // op_type
    else if (f == 5 && w == 2) {
      OnnxAttr a = parse_attr(r.bytes());
      n.attr[a.name] = a;
    } // attribute
    else
      r.skip(w);
  }
  return n;
}

// ValueInfoProto -> {name, shape}.  Shape is nested:
//   ValueInfoProto.type(2) -> TypeProto.tensor_type(1) -> Tensor.shape(2)
//   -> TensorShapeProto.dim(1, repeated) -> Dimension.dim_value(1)
inline void parse_valueinfo(const std::string &b, std::string &name, std::vector<int64_t> &shape) {
  PB r(b.data(), b.data() + b.size());
  int f, w;
  while (!r.done()) {
    r.tag(f, w);
    if (f == 1 && w == 2)
      name = r.bytes();
    else if (f == 2 && w == 2) { // type
      std::string ty = r.bytes();
      PB t(ty.data(), ty.data() + ty.size());
      int tf, tw;
      while (!t.done()) {
        t.tag(tf, tw);
        if (tf == 1 && tw == 2) { // tensor_type
          std::string tt = t.bytes();
          PB tr(tt.data(), tt.data() + tt.size());
          int sf, sw;
          while (!tr.done()) {
            tr.tag(sf, sw);
            if (sf == 2 && sw == 2) { // shape
              std::string sh = tr.bytes();
              PB sr(sh.data(), sh.data() + sh.size());
              int df, dw;
              while (!sr.done()) {
                sr.tag(df, dw);
                if (df == 1 && dw == 2) { // dim
                  std::string dm = sr.bytes();
                  PB dr(dm.data(), dm.data() + dm.size());
                  int xf, xw;
                  int64_t dv = 0;
                  while (!dr.done()) {
                    dr.tag(xf, xw);
                    if (xf == 1 && xw == 0)
                      dv = (int64_t)dr.varint();
                    else
                      dr.skip(xw);
                  }
                  shape.push_back(dv);
                } else
                  sr.skip(dw);
              }
            } else
              tr.skip(sw);
          }
        } else
          t.skip(tw);
      }
    } else
      r.skip(w);
  }
}

inline OnnxGraph parse_graph(const std::string &b) {
  OnnxGraph g;
  PB r(b.data(), b.data() + b.size());
  int f, w;
  while (!r.done()) {
    r.tag(f, w);
    if (f == 1 && w == 2)
      g.nodes.push_back(parse_node(r.bytes())); // node
    else if (f == 5 && w == 2) {
      OnnxTensor t = parse_tensor(r.bytes());
      g.initializer[t.name] = std::move(t);
    } // initializer
    else if (f == 11 && w == 2) {
      std::string nm;
      std::vector<int64_t> sh;
      parse_valueinfo(r.bytes(), nm, sh);
      g.inputs.push_back(nm);
      g.input_shapes[nm] = sh;
    } // input
    else if (f == 12 && w == 2) {
      std::string nm;
      std::vector<int64_t> sh;
      parse_valueinfo(r.bytes(), nm, sh);
      g.outputs.push_back(nm);
    } // output
    else
      r.skip(w);
  }
  return g;
}

inline OnnxGraph load_onnx(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("onnx: cannot open " + path);
  std::string buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  PB r(buf.data(), buf.data() + buf.size());
  int f, w;
  OnnxGraph g;
  bool found = false;
  while (!r.done()) {
    r.tag(f, w);
    if (f == 7 && w == 2) { g = parse_graph(r.bytes()); found = true; break; }
    r.skip(w);
  } // ModelProto.graph
  if (!found) throw std::runtime_error("onnx: no graph found");
  // resolve external-data weights (large LLMs store initializers in a sibling .data file)
  auto slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
  // Read each tensor's slice DIRECTLY from the .data file (seek+read) rather than slurping the whole
  // file into a string blob.  A multi-GB LLM .data held as a blob AND decoded into fp32 vectors is a
  // 2x (blob + decoded) transient that dominates peak RSS -- Llama's 6.3 GB .data was a 12.6 GB load
  // spike.  Per-slice reads hold only ONE tensor's bytes at a time on top of the decoded weights.
  std::map<std::string, std::ifstream> fcache; // keep each .data file open, seek per tensor
  std::map<std::string, long long> fsz_cache;
  int _next = 0, _ntot = (int)g.initializer.size(); long long _nb = 0;
  for (auto &kv : g.initializer) {
    OnnxTensor &t = kv.second;
    if (!t.ext) continue;
    ++_next; _nb += t.ext_len;
    const std::string fp = dir + t.ext_loc;
    auto it = fcache.find(fp);
    if (it == fcache.end()) {
      std::ifstream ef(fp, std::ios::binary);
      if (!ef) throw std::runtime_error("onnx: cannot open external data " + fp);
      ef.seekg(0, std::ios::end); fsz_cache[fp] = (long long)ef.tellg();
      it = fcache.emplace(fp, std::move(ef)).first;
    }
    std::ifstream &ef = it->second;
    const long long fsz = fsz_cache[fp];
    long long len = t.ext_len > 0 ? t.ext_len : fsz - t.ext_off;
    if (t.ext_off < 0 || t.ext_off + len > fsz)
      throw std::runtime_error("onnx: external-data slice OOB for " + t.name);
    t.ext_path = fp;                 // remember source so it can be (re)read on demand
    if (getenv("LAZY_INIT")) continue; // decode lazily in the graph runner (resolve_ext); keep data empty
    std::string slice((size_t)len, '\0');
    ef.seekg((std::streamoff)t.ext_off);
    ef.read(&slice[0], (std::streamsize)len);
    decode_raw(t, slice); // slice freed at loop-iteration end
  }
  if (getenv("DBG_ONNX")) {
    printf("[onnx] %s: %d/%d initializers external, %.1f MB resolved\n", path.c_str(), _next, _ntot, _nb / 1e6);
    fflush(stdout);
  }
  return g;
}

} // namespace volefier
#endif // ML_IO_ONNX_H
