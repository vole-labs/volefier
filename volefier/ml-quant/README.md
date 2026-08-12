# ml-quant — per-tensor quantized inference

This directory proves neural-network inference using **per-tensor
quantization**, the scheme used by recent lookup-based systems (zkGPT,
DeepProve).

Every tensor of activations is represented as integers under its own scale
factor, so the values throughout the network are small integers rather than real
numbers. A linear layer (a matrix multiply or convolution) takes inputs and
weights at their respective scales, accumulates the integer products, and then
rescales the result down to the scale chosen for its output. Because each tensor
may have a different scale, the rescale factor is a tensor-specific constant
rather than a single fixed power of two — this is what makes it true
quantization rather than plain fixed-point.

The whole computation stays in the prover's arithmetic field. The rescaling
after each layer, and the comparisons needed for activations such as ReLU and
max-pooling, are proven with **lookup-based range checks** instead of decomposing
values into bits — so there is no sign-bit extraction and no excursion into
Boolean circuits. The smooth non-linearities (exponentials for softmax, the GeLU
curve) are evaluated through small lookup tables, and normalization layers avoid
an explicit square root by committing to the inverse standard deviation and
checking its consistency with range proofs.

The directory covers both families of models: transformer language models
(token embeddings, multi-head attention, normalization, the feed-forward block,
and a final classification head that certifies the predicted token) and
convolutional networks (the standard residual and bottleneck building blocks
assembled into the common ResNet and VGG depths).

Because nothing ever leaves the field — no bit decomposition, no floating-point
circuits — this is the fast path. The linear algebra itself is a negligible
fraction of the proving cost; the lookups dominate but are individually cheap.
The companion directories implement the same models under a single uniform scale
([`../ml-fixed`](../ml-fixed)) and under exact floating point
([`../ml-float`](../ml-float)) for comparison.
