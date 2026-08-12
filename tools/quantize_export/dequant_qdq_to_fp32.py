import sys, numpy as np, onnx
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
new_init = {}            # name -> np array (fp32 folded weights)
rename = {}              # activation Q/DQ identity rewire: output -> source
def root(n):
    while n in rename: n = rename[n]
    return n
keep = []
for nd in g.node:
    op = nd.op_type; I = list(nd.input); O = nd.output[0]
    if op == "DequantizeLinear":
        s = init[I[1]]; z = init[I[2]] if len(I) > 2 else np.array(0)
        if I[0] in init:                              # weight/bias constant -> fold to fp32
            w = init[I[0]].astype(np.float64)
            sc = np.asarray(s, np.float64); zp = np.asarray(z, np.float64)
            if sc.ndim == 1 and w.ndim > 1:           # per-channel along axis 0 (conv/gemm out-dim)
                shp = [w.shape[0]] + [1]*(w.ndim-1); sc = sc.reshape(shp); zp = zp.reshape(shp)
            new_init[O] = ((w - zp) * sc).astype(np.float32)
        else:                                         # activation DQ -> identity
            rename[O] = I[0]
    elif op == "QuantizeLinear":
        rename[O] = I[0]                              # activation Q -> identity
    else:
        keep.append(nd)
# rebuild nodes, rewiring inputs through rename roots
nodes = []
for nd in keep:
    ins = [root(x) if x else x for x in nd.input]
    nodes.append(helper.make_node(nd.op_type, ins, list(nd.output), nd.name,
                                  **{a.name: helper.get_attribute_value(a) for a in nd.attribute}))
# initializers: keep non-quant constants that are still referenced as fp32 + folded weights
inits = []
for t in g.initializer:
    arr = init[t.name]
    if arr.dtype in (np.int8, np.uint8, np.int32) and t.name not in new_init:
        continue                                     # drop raw int quant params (scales/zps/int weights)
    inits.append(numpy_helper.from_array(arr.astype(np.float32) if arr.dtype==np.float64 else arr, t.name))
for nm, arr in new_init.items():
    inits.append(numpy_helper.from_array(arr, nm))
outs = [helper.make_tensor_value_info(root(o.name), onnx.TensorProto.FLOAT, None) for o in g.output]
ins  = [i for i in g.input if i.name not in init]
ng = helper.make_graph(nodes, g.name, ins, outs, inits)
om = helper.make_model(ng, opset_imports=m.opset_import)
om.ir_version = m.ir_version
onnx.save(om, dst)
print("wrote", dst, "nodes:", len(nodes))
