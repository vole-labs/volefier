#!/usr/bin/env python3
# macOS counterpart of throttle.py (Linux tc tbf+netem).
#
# Linux: `tc qdisc tbf` (bandwidth) + `netem delay` (latency), applied to the
# EGRESS of a physical interface.
# macOS: there is no tc. The equivalent stack is dummynet + pf:
#   - `dnctl pipe N config bw <B>Mbit/s delay <L>ms`   <- the tbf+netem of macOS
#   - a pf rule `dummynet out ... pipe N`              <- routes traffic into it
# Like the Linux script, this shapes OUTBOUND (egress) traffic only, so the
# one-way delay added equals -l and (with both machines throttled) RTT ~= 2*L,
# matching the tc setup's semantics.
#
# Usage mirrors throttle.py:
#   sudo python3 throttle_macos.py -i auto -b 100 -l 20     # 100 Mbit/s, 20 ms egress delay
#   sudo python3 throttle_macos.py -i en0  -b 50            # 50 Mbit/s, no added delay
#   sudo python3 throttle_macos.py -i auto -d               # remove throttle
#
# Notes:
#  * Must run as root (dnctl/pfctl need it) -- prefix with sudo.
#  * dummynet `delay` is ONE-WAY, same as netem; quantized to the kernel timer.
#  * Shapes only the chosen interface; use the real NIC (en0/en7/...) for a
#    two-machine benchmark, or lo0 for a localhost two-party run.

import subprocess
import argparse
import sys

PIPE = "1"          # dummynet pipe number
ANCHOR = "throttle"  # pf anchor name we load our rule into
PF_CONF = "/etc/pf.conf"


def find_dev():
    """Default-route interface, the macOS analogue of throttle.py's find_dev()."""
    o = subprocess.check_output(["route", "-n", "get", "default"]).decode()
    for line in o.splitlines():
        line = line.strip()
        if line.startswith("interface:"):
            return line.split(":", 1)[1].strip()
    print("Could not determine default interface; pass -i explicitly.")
    sys.exit(1)


def sh(cmd):
    print(cmd)
    return subprocess.call(["bash", "-c", cmd])


def add_throttle(dev, bw, delay):
    # 1) Configure the dummynet pipe (bandwidth + one-way delay).
    sh("dnctl pipe {p} config bw {b}Mbit/s delay {d}ms".format(p=PIPE, b=bw, d=delay))
    # 2) Make the main ruleset reference our anchor (append to the system pf.conf
    #    so existing rules are preserved), then load the dummynet rule into it.
    sh("( cat {conf}; printf 'dummynet-anchor \"{a}\"\\nanchor \"{a}\"\\n' ) "
       "| pfctl -f -".format(conf=PF_CONF, a=ANCHOR))
    sh("echo 'dummynet out on {dev} proto tcp from any to any pipe {p}' "
       "| pfctl -a {a} -f -".format(dev=dev, p=PIPE, a=ANCHOR))
    # 3) Enable pf (idempotent; -E ref-counts the enable).
    sh("pfctl -E")


def remove_throttle(dev):
    sh("pfctl -a {a} -F all".format(a=ANCHOR))  # flush our anchor's rules
    sh("dnctl -q flush")                        # delete all dummynet pipes
    sh("pfctl -f {conf}".format(conf=PF_CONF))  # restore the stock ruleset
    sh("pfctl -d")                              # disable pf


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--interface", action="store", dest="i", type=str, required=True)
parser.add_argument("-b", "--bandwith", action="store", dest="b", type=int)
parser.add_argument("-l", "--latency", action="store", dest="l", type=int)
parser.add_argument("-d", "--delete", action="store_true")
args = parser.parse_args()

DEV = find_dev() if args.i == "auto" else args.i
print(vars(args))

if args.delete:
    remove_throttle(DEV)
else:
    if args.b is not None:
        add_throttle(DEV, args.b, args.l if args.l is not None else 0)
    else:
        print("error! -b/--bandwith is required (or use -d to delete)")
        sys.exit(1)
