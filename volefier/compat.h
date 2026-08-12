// Compatibility shim: adapt volefier's call sites to the updated emp-toolkit.
//
// The new emp-toolkit changed: BoolIO is no longer a template (BoolIO<NetIO> -> BoolIO),
// setup/finalize are non-template and single-io (setup_zk_arith(io, party)), helpers
// parse_party_and_port / int_to_bool / fix_key were removed, fp_zkp_inner_prdt / ZKFpExec*
// are non-template, and the IO byte counter split into send_counter + recv_counter.
//
// This header restores the few removed helpers and a multi-io setup overload so the bulk
// of volefier compiles after a mechanical type rename.  Include AFTER emp-tool/emp-zk.
#ifndef EMP_ZKML_COMPAT_H
#define EMP_ZKML_COMPAT_H

#include <cstdlib>

namespace emp {

// old: parse_party_and_port(argv, &party, &port) -- party from argv[1], port from argv[2].
inline void parse_party_and_port(char **arg, int *party, int *port) {
  *party = parse_party(arg);
  *port = (arg[1] != nullptr && arg[2] != nullptr) ? atoi(arg[2]) : peer_port();
}

// old multi-io arith setup -> new single-io setup with a thread count. The updated emp-zk
// threads the sVOLE generation internally via setup_zk_arith(io, party, threads), so forward
// the old `threads` arg to drive the parallel VOLE / commitment.
inline void setup_zk_arith(BoolIO **ios, int threads, int party, int64_t expected_vole = 0,
                           BoolIO *vole_io = nullptr, int vole_threads = -1) {
  setup_zk_arith(ios[0], party, threads, expected_vole, vole_io, vole_threads);   // expected_vole>0 -> prepay VOLE; vole_io!=null -> background sVOLE on a 2nd socket; vole_threads<0 -> = threads
}

// Total bytes over a channel (the old single `counter` is now send + recv).
inline uint64_t io_counter(IOChannel *io) { return io->send_counter + io->recv_counter; }

// old: fix_key was a fixed 16-byte AES key in emp-tool/utils/constants.h; removed in the new emp.
// Restore it so callers can seed a deterministic PRG via PRG(fix_key). (const -> internal linkage.)
inline const char fix_key[] = "\x61\x7e\x8d\xa2\xa0\x51\x1e\x96\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";

}  // namespace emp

#endif  // EMP_ZKML_COMPAT_H
