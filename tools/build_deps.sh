#!/bin/bash
# Build the emp submodules (external/{emp-tool,emp-ot,emp-zk}) into external/install.
# No sudo, no global installs: configure emp-zkml with
#   cmake -B build -DCMAKE_PREFIX_PATH="$PWD/external/install" ...
# Fresh checkout:  git submodule update --init --recursive && tools/build_deps.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$ROOT/external/install"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
EXTRA=()
if [ "$(uname)" = "Darwin" ]; then
  OSSL=$(ls -d /opt/homebrew/opt/openssl@3* 2>/dev/null | sort -V | tail -1)
  [ -n "$OSSL" ] && EXTRA+=("-DOPENSSL_ROOT_DIR=$OSSL")
fi
for dep in emp-tool emp-ot emp-zk; do            # dependency order: tool -> ot -> zk
  echo "=== $dep ==="
  cmake -S "$ROOT/external/$dep" -B "$ROOT/external/$dep/build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_PREFIX_PATH="$PREFIX" "${EXTRA[@]}"
  cmake --build "$ROOT/external/$dep/build" -j"$JOBS"
  cmake --install "$ROOT/external/$dep/build"
done
echo "== deps installed to $PREFIX =="
