# ml-float — exact floating-point inference (Mystique approach)

This directory proves neural-network inference the way the Mystique system does:
the **linear layers run in the fixed-point field**, while the **non-linear parts
are computed in genuine IEEE-754 floating-point arithmetic**. The motivation is
accuracy — floating point reproduces the original model's behavior exactly,
without the small errors introduced by quantization or by approximating curves
with lookup tables. The price is that floating-point and bit-level operations are
expensive to prove, so this is the accurate-but-slow path, in deliberate
contrast to the field-only lookup tracks in
[`../ml-quant`](../ml-quant) and [`../ml-fixed`](../ml-fixed).

Because two different worlds are involved — field arithmetic for the linear
algebra and bit/floating-point logic for everything else — activations have to be
converted back and forth as they pass between layers. The underlying ZK library
provides this bridge: it can move a value between the arithmetic field and a
bit-level representation, and between fixed-point and floating-point form, all
within a single proof. The cost of this conversion, rather than the arithmetic
itself, turns out to dominate everything.

The directory contains three things:

- A set of **standalone floating-point non-linear functions** — the activation,
  normalization, and softmax operations evaluated exactly in floating point.
  These are the float analogues of the lookup-table versions in the other tracks.

- A **first, float-native network**, where activations are carried in floating
  point throughout and are converted into the field only to run each convolution.
  This was the naive starting point and is kept for comparison.

- A **graph-driven ONNX front end** that runs a whole fp32 model on this backend
  and follows Mystique faithfully — field linear layers, Boolean ReLU and pooling,
  and floating-point BatchNorm/Sigmoid/GeLU — converting between the field and the
  bit/float world at each layer boundary. Activations live in the fixed-point field,
  and floating point is used only where it is genuinely required (the per-channel
  square root inside batch normalization); truncating back to the working scale
  after each multiply is a field range check rather than a floating-point
  round-trip. These choices are what make it far cheaper than the naive network.
  This is the current entry point for real models.

The lesson from this directory is concrete: almost all of the cost is the
crossing between the arithmetic field and the bit/floating-point world — the
truncations and comparisons — not the matrix multiplies and not the float math
as such. That crossing is precisely what the lookup-based tracks avoid by doing
their rescales and comparisons with field range checks, and it is the
order-of-magnitude difference between the floating-point generation of provers
and the lookup generation.

This track uses the library's Boolean proof backend in addition to the
arithmetic one, which on first run needs a local scratch directory for its
one-time setup data.
