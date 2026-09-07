#!/bin/bash
# Build the ZK back-end (external/vole-zk, vole-labs/vole-zk) into external/install.
# vole-zk vendors vole-labs/vole, which in turn vendors emp-tool and emp-ot; all are
# built in-tree from their pinned submodule commits, so nothing else is installed.
# No sudo, no global installs: configure volefier with
#   cmake -B build -DCMAKE_PREFIX_PATH="$PWD/external/install" ...
# Fresh checkout:  git submodule update --init --recursive && tools/build_deps.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$ROOT/external/install"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
DEP="$ROOT/external/vole-zk"
[ -f "$DEP/thirdparty/vole/thirdparty/emp-tool/CMakeLists.txt" ] || {
  echo "external/vole-zk submodules missing; run: git submodule update --init --recursive" >&2; exit 1; }
EXTRA=()
if [ "$(uname)" = "Darwin" ]; then
  OSSL=$(ls -d /opt/homebrew/opt/openssl@3* 2>/dev/null | sort -V | tail -1)
  [ -n "$OSSL" ] && EXTRA+=("-DOPENSSL_ROOT_DIR=$OSSL")
fi
echo "=== vole-zk (vole + emp-tool + emp-ot in-tree) ==="
cmake -S "$DEP" -B "$DEP/build" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DEMP_ZK_BUILD_TESTS=OFF "${EXTRA[@]}"
cmake --build "$DEP/build" -j"$JOBS"
cmake --install "$DEP/build"
echo "== deps installed to $PREFIX =="
