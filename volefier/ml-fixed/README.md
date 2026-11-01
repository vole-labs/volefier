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
backward; attention, feed-forward and LM-head backward; and SGD or AdamW weight
updates), so a full forward-and-backward step is one continuous proof. It supports
**LoRA fine-tuning** in particular (`LoRAAdapter`, `LoRABlockBackward`): the
effective weight `W + (alpha/r)·B·A` is proven from the committed adapters, so the
forward pass is bound to them; the base model and the LoRA init are public, so the
statement names the model being tuned; labels are private (committed one-hot); the
adapters keep extra fractional bits so small steps do not round away; the backward
for frozen weights works on the whole sequence and produces gradients only for the
adapted projections; and several optimizer steps chain inside one proof. Each step
runs a mini-batch (`BATCH` sequences, their gradients summed, one update on the mean
loss) and updates the adapters by SGD or AdamW (`OPT=adamw`, `FixedAdamW`: committed
first/second-moment state carried across steps, bias correction and decoupled weight
decay, the square root and division proven with bounded relations, and a master copy of
the adapters 16 bits finer so that the per-step decay and small updates are not rounded
away). The training
forward (`forward_seq`) does every matmul over the whole sequence, as inference does.
By default the model is random; `GPT2_BIN`/`TOKENS` load real GPT-2 small and a
WikiText-2 window (`BATCH * SEQ + 1` tokens) exported by
[`tools/quantize_export/export_gpt2_lora.py`](../../tools/quantize_export/export_gpt2_lora.py)
(tokens committed one-hot, embeddings proven as `onehot @ wte + wpe`, `ln_f` before the
tied head). See [`test/ml-fixed/lora_train.h`](../../test/ml-fixed/lora_train.h). All of this
reuses the same field machinery — the batched inner-product and Freivalds matmuls
and the bounded requant — adding no new proof primitive beyond the forward
pass except one lookup table for the GeLU derivative.

The fixed-point number format used here (a Mersenne-prime field, a fixed
fractional precision, and a value range kept small enough that a single layer's
multiply cannot overflow) is the same encoding the floating-point track
([`../ml-float`](../ml-float)) uses for its linear layers — the difference being
that this track also proves the non-linearities with field lookups, whereas the
float track computes them in floating-point circuits.
