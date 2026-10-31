// Test: ml-fixed single-block LoRA fine-tuning proof at a toy shape (2 SGD steps, one ZK session).
// The proof and its checks live in lora_train.h; this file only sets the default shape.
#include "lora_train.h"

int main(int argc, char **argv) {
  // NB, SEQ, D, HEADS, FFN, VOCAB, RANK, STEPS
  return lora_train_main(argc, argv, {1, 3, 4, 2, 8, 8, 2, 2}, "lora_train_step");
}
