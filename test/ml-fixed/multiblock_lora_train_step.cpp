// Test: ml-fixed multi-block LoRA fine-tuning proof at GPT-2-small shape (12 blocks, seq 512,
// d 768, 12 heads, ffn 3072, vocab 50257, rank 8), 2 optimizer steps in one ZK session.
// The proof and its checks live in lora_train.h; every dimension can be overridden from the env.
#include "lora_train.h"

int main(int argc, char **argv) {
  // NB, SEQ, D, HEADS, FFN, VOCAB, RANK, STEPS
  return lora_train_main(argc, argv, {12, 512, 768, 12, 3072, 50257, 8, 2}, "multiblock_lora_train_step");
}
