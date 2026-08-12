# ml-fixed — uniform fixed-point inference

This directory proves the same neural-network inference as the quantized track,
but using **uniform fixed-point** arithmetic: instead of giving every tensor its
own scale, all weights and activations across the whole network share a single
fixed scale (a fixed number of fractional bits). Real values are represented as
integers by multiplying by that one common factor.

Everything else is the same as the quantized track. Linear layers accumulate
integer products and rescale afterward, but here the rescale is always the same
plain power-of-two shift rather than a tensor-specific factor. Comparisons for
activations and the rescaling are proven with lookup-based range checks entirely
inside the field, the smooth non-linearities use lookup tables, and the
normalization layers avoid an explicit square root the same way.

The difference between this track and the quantized one is **representational,
not cryptographic**. Choosing a single uniform scale versus per-tensor scales is
a question of numerical range and accuracy; it does not change the underlying
proof machinery, and the proving cost is essentially identical. The two are kept
side by side so the accuracy and cost of uniform fixed-point can be compared
directly against per-tensor quantization.

It covers the same model families — transformer language models and the standard
residual and convolutional networks at the usual depths — and, like the
quantized track, keeps the entire computation in the field with no bit
decomposition and no floating-point circuits.

Beyond inference, this track also proves a **training step** — a
proof-of-fine-tuning. Backward-pass gadgets consume the committed forward-pass
activations and produce committed gradients (softmax, GeLU, LayerNorm and RMSNorm
backward; attention, feed-forward and LM-head backward; and an SGD weight
update), so a full forward-and-backward step is one continuous proof. It supports
**LoRA fine-tuning** in particular: the adapted projections use an effective
weight committed as a witness, the block backward yields that weight's gradient,
and only the low-rank adapters are updated while the base model stays frozen. All
of this reuses the same field machinery — the batched inner-product matmul (now
also for its two backward transposes) and the two-sided-range requant — adding no
new proof primitive beyond the forward pass except one lookup table for the GeLU
derivative.

The fixed-point number format used here (a Mersenne-prime field, a fixed
fractional precision, and a value range kept small enough that a single layer's
multiply cannot overflow) is the same encoding the floating-point track
([`../ml-float`](../ml-float)) uses for its linear layers — the difference being
that this track also proves the non-linearities with field lookups, whereas the
float track computes them in floating-point circuits.
