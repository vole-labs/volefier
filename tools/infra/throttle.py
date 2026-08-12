#!/usr/python
import subprocess
def find_dev():
    l = []
    o = subprocess.check_output(['ifconfig'])
    for i in o.split("\n"):
        if "flags" in i:
            l.append(i.split(":")[0])
    res = ""
    for i in l:
        if i != "lo":
            res = i
    if len(l) > 2:
        print("More than two candidates!")
    return res
remove_throttle = '''
sudo tc qdisc del dev DEV root
'''

add_throttle = '''
sudo tc qdisc del dev DEV root
sudo tc qdisc add dev DEV root handle 1: tbf rate BANDWIDTHmbit burst 100000 limit 10000
sudo tc qdisc add dev DEV parent 1:1 handle 10: netem delay DELAYmsec
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--interface', action='store', dest='i', type = str, required = True)
parser.add_argument('-b', '--bandwith', action='store', dest = 'b', type = int)
parser.add_argument('-l', '--latency', action='store', dest = 'l', type = int)
parser.add_argument('-d', '--delete', action='store_true')
args = parser.parse_args()
DEV = ""
if args.i == "auto":
    DEV = find_dev()
else:
    DEV = args.i
print(vars(args))
if vars(args)['delete']:
    subprocess.call(["bash", "-c", remove_throttle.replace("DEV", DEV)])
else:
    if vars(args)['b']!=None and vars(args)['l'] != None :
        cmd = add_throttle.replace("DEV", DEV).replace("BANDWIDTH", str(vars(args)['b']))\
                                                   .replace("DELAY", str(vars(args)['l']))
        print(cmd)
        subprocess.call(["bash", "-c", cmd])
    elif vars(args)['b']!=None and vars(args)['l'] == None :
        cmd = add_throttle.replace("DEV", DEV).replace("BANDWIDTH", str(vars(args)['b']))\
                                                   .replace("DELAY", "0")
        print(cmd)
        subprocess.call(["bash", "-c", cmd])
    else:
        print("error!")
