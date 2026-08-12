#!/usr/bin/env bash
# Sweep bench_logup across (logN, W) configurations.
# The binary launches both ALICE and BOB on the same machine over loopback,
# so each (logN, W) run gets a fresh setup_zk_arith / finalize_zk_arith
# cycle — no cache or VOLE-pool sharing across configurations.
#
# Run from the build directory:
#   cd build
#   bash ../test/ram/bench_sweep.sh
#
# Optional args (positional):
#   $1 = port      (default 12345)
#   $2 = ALICE IP  (default 127.0.0.1)
#   $3 = output dir (default ./bench_results)
#
# Per-run logs land in $OUTDIR/<binary>_logN<logN>_W<W>.{alice,bob};
# a final summary table is printed to stdout and saved to $OUTDIR/summary.txt.

set -euo pipefail

PORT="${1:-12345}"
IP="${2:-127.0.0.1}"
OUTDIR="${3:-./bench_results}"

# Sweep dimensions — edit here if you want different points
LOG_NS=(8 12 16)
WS=(1048576 8388608 67108864)   # 2^20, 2^23, 2^26

mkdir -p "$OUTDIR"

run_pair() {
    local bin="$1" logN="$2" W="$3"
    local tag="${bin}_logN${logN}_W${W}"

    # Launch BOB first (listens), give it a moment, then ALICE dials in.
    "./bin/$bin" 2 "$PORT" "$logN" "$W"      > "$OUTDIR/${tag}.bob"   2>&1 &
    local pid_b=$!
    sleep 0.5
    "./bin/$bin" 1 "$PORT" "$logN" "$W" "$IP" > "$OUTDIR/${tag}.alice" 2>&1 &
    local pid_a=$!

    # Wait for both — `|| true` so a failed run doesn't abort the whole sweep
    wait "$pid_a" 2>/dev/null || true
    wait "$pid_b" 2>/dev/null || true
}

echo "Sweep:  logN ∈ {${LOG_NS[*]}}  ×  W ∈ {${WS[*]}}"
echo "Output: $OUTDIR"
echo

for logN in "${LOG_NS[@]}"; do
    for W in "${WS[@]}"; do
        echo "[run] logN=$logN W=$W"
        run_pair test_ram_bench_logup "$logN" "$W"
    done
done

echo
SUMMARY="$OUTDIR/summary.txt"
{
    echo "==================== summary (ALICE side) ===================="
    printf "%-7s %5s %12s %12s %16s %12s\n" \
        "variant" "logN" "W" "time_s" "bytes" "us/access"
    echo "-------------------------------------------------------------------"
    for logN in "${LOG_NS[@]}"; do
        for W in "${WS[@]}"; do
            f="$OUTDIR/test_ram_bench_logup_logN${logN}_W${W}.alice"
            line="$(grep -h '^\[RESULT\]' "$f" 2>/dev/null || true)"
            time_us="$(echo "$line" | sed -nE 's/.*time_us=([0-9]+).*/\1/p')"
            bytes="$(echo  "$line" | sed -nE 's/.*bytes=([0-9]+).*/\1/p')"
            if [ -n "${time_us:-}" ] && [ -n "${bytes:-}" ]; then
                time_s=$(awk -v t="$time_us"        'BEGIN{printf "%.2f", t/1e6}')
                per_us=$(awk -v t="$time_us" -v w="$W" 'BEGIN{printf "%.3f", t/w}')
                printf "%-7s %5s %12s %12s %16s %12s\n" \
                    "logup" "$logN" "$W" "$time_s" "$bytes" "$per_us"
            else
                printf "%-7s %5s %12s %12s %16s %12s\n" \
                    "logup" "$logN" "$W" "?" "?" "?"
            fi
        done
    done
} | tee "$SUMMARY"

echo
echo "Saved summary to $SUMMARY"
