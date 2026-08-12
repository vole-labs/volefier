#ifndef ML_IO_SAFETENSORS_H
#define ML_IO_SAFETENSORS_H

// Minimal, dependency-free SafeTensors reader.
//
// File layout:  [u64 header_len][JSON header (header_len bytes)][raw tensor data]
// The JSON header maps each tensor name to {dtype, shape, data_offsets:[begin,end]},
// with offsets relative to the start of the data region. Reads F32 / F16 / BF16
// weights (the common float checkpoint dtypes) into row-major float tensors.
//
// This is the format-level loader; mapping the named tensors onto a specific
// model architecture and quantizing them is done by per-model code on top of it.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace volefier {

struct STensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<float> data; // row-major, decoded to float
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

// ---- tiny JSON value (object / array / string / number) ----
struct STJson {
  enum { OBJ, ARR, STR, NUM } type = OBJ;
  std::map<std::string, STJson> obj;
  std::vector<STJson> arr;
  std::string str;
  double num = 0;
};

class STJsonParser {
public:
  STJsonParser(const char *p, const char *e) : p_(p), e_(e) {}
  STJson parse() {
    ws();
    return value();
  }

private:
  const char *p_, *e_;
  void ws() {
    while (p_ < e_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
  }
  STJson value() {
    ws();
    if (p_ >= e_) throw std::runtime_error("safetensors json: unexpected end");
    if (*p_ == '{') return object();
    if (*p_ == '[') return array();
    if (*p_ == '"') {
      STJson v;
      v.type = STJson::STR;
      v.str = string();
      return v;
    }
    STJson v;
    v.type = STJson::NUM;
    v.num = number();
    return v;
  }
  STJson object() {
    STJson v;
    v.type = STJson::OBJ;
    ++p_;
    ws();
    if (p_ < e_ && *p_ == '}') {
      ++p_;
      return v;
    }
    while (true) {
      ws();
      std::string k = string();
      ws();
      if (p_ >= e_ || *p_ != ':') throw std::runtime_error("safetensors json: expected ':'");
      ++p_;
      v.obj[k] = value();
      ws();
      if (p_ < e_ && *p_ == ',') {
        ++p_;
        continue;
      }
      if (p_ < e_ && *p_ == '}') {
        ++p_;
        break;
      }
      throw std::runtime_error("safetensors json: expected ',' or '}'");
    }
    return v;
  }
  STJson array() {
    STJson v;
    v.type = STJson::ARR;
    ++p_;
    ws();
    if (p_ < e_ && *p_ == ']') {
      ++p_;
      return v;
    }
    while (true) {
      v.arr.push_back(value());
      ws();
      if (p_ < e_ && *p_ == ',') {
        ++p_;
        continue;
      }
      if (p_ < e_ && *p_ == ']') {
        ++p_;
        break;
      }
      throw std::runtime_error("safetensors json: expected ',' or ']'");
    }
    return v;
  }
  std::string string() {
    if (p_ >= e_ || *p_ != '"') throw std::runtime_error("safetensors json: expected string");
    ++p_;
    std::string s;
    while (p_ < e_ && *p_ != '"') {
      if (*p_ == '\\' && p_ + 1 < e_) {
        ++p_;
        s.push_back(*p_);
      } // SafeTensors names rarely escape; keep raw
      else
        s.push_back(*p_);
      ++p_;
    }
    if (p_ >= e_) throw std::runtime_error("safetensors json: unterminated string");
    ++p_;
    return s;
  }
  double number() {
    const char *s = p_;
    while (p_ < e_ && (*p_ == '-' || *p_ == '+' || *p_ == '.' || *p_ == 'e' || *p_ == 'E' ||
                       (*p_ >= '0' && *p_ <= '9')))
      ++p_;
    return std::stod(std::string(s, p_));
  }
};

class SafeTensors {
public:
  std::map<std::string, STensor> tensors;

  explicit SafeTensors(const std::string &path) {
    load(path);
  }

  const STensor &at(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("safetensors: tensor not found: " + name);
    return it->second;
  }
  bool has(const std::string &name) const {
    return tensors.count(name) > 0;
  }

private:
  static float bf16_to_float(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float o;
    std::memcpy(&o, &f, 4);
    return o;
  }
  static float f16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
    if (exp == 0) {
      if (mant == 0)
        f = sign;
      else {
        exp = 1;
        while (!(mant & 0x400)) {
          mant <<= 1;
          --exp;
        }
        mant &= 0x3FF;
        f = sign | ((exp + 112) << 23) | (mant << 13);
      }
    } else if (exp == 0x1F)
      f = sign | 0x7F800000u | (mant << 13);
    else
      f = sign | ((exp + 112) << 23) | (mant << 13);
    float o;
    std::memcpy(&o, &f, 4);
    return o;
  }

  void load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char *>(&hlen), 8);
    if (!f || hlen == 0 || hlen > (1ull << 32))
      throw std::runtime_error("safetensors: bad header length");
    std::string header(hlen, '\0');
    f.read(&header[0], (std::streamsize)hlen);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    STJson root = STJsonParser(header.data(), header.data() + header.size()).parse();
    for (auto &kv : root.obj) {
      if (kv.first == "__metadata__") continue;
      const STJson &t = kv.second;
      STensor st;
      st.dtype = t.obj.at("dtype").str;
      for (auto &d : t.obj.at("shape").arr) st.shape.push_back((int64_t)d.num);
      const auto &off = t.obj.at("data_offsets").arr;
      int64_t b = (int64_t)off[0].num, e = (int64_t)off[1].num;
      if (e > (int64_t)blob.size() || b > e)
        throw std::runtime_error("safetensors: offsets out of range for " + kv.first);
      const char *raw = blob.data() + b;
      int64_t nbytes = e - b, n = st.numel();
      st.data.resize(n);
      if (st.dtype == "F32") {
        if (nbytes != 4 * n) throw std::runtime_error("F32 size mismatch");
        std::memcpy(st.data.data(), raw, nbytes);
      } else if (st.dtype == "F16") {
        for (int64_t i = 0; i < n; ++i) {
          uint16_t h;
          std::memcpy(&h, raw + 2 * i, 2);
          st.data[i] = f16_to_float(h);
        }
      } else if (st.dtype == "BF16") {
        for (int64_t i = 0; i < n; ++i) {
          uint16_t h;
          std::memcpy(&h, raw + 2 * i, 2);
          st.data[i] = bf16_to_float(h);
        }
      } else
        throw std::runtime_error("safetensors: unsupported dtype " + st.dtype +
                                 " (only F32/F16/BF16)");
      tensors[kv.first] = std::move(st);
    }
  }
};

// Write a set of named float tensors as a SafeTensors file (F32). Tensors are
// laid out in name order; offsets are assigned sequentially.
inline void save_safetensors(const std::string &path,
                             const std::map<std::string, STensor> &tensors) {
  std::string data, header = "{";
  bool first = true;
  int64_t off = 0;
  for (const auto &kv : tensors) {
    const STensor &t = kv.second;
    int64_t nb = t.numel() * 4;
    header += (first ? "" : ",");
    first = false;
    header += "\"" + kv.first + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(t.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nb) + "]}";
    for (float v : t.data) {
      uint32_t b;
      std::memcpy(&b, &v, 4);
      for (int i = 0; i < 4; ++i) data.push_back((char)((b >> (8 * i)) & 0xff));
    }
    off += nb;
  }
  header += "}";
  std::ofstream o(path, std::ios::binary);
  uint64_t hlen = header.size();
  o.write(reinterpret_cast<const char *>(&hlen), 8);
  o.write(header.data(), (std::streamsize)header.size());
  o.write(data.data(), (std::streamsize)data.size());
}

} // namespace volefier

#endif // ML_IO_SAFETENSORS_H
