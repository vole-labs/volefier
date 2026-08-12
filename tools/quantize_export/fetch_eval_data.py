# Download the GPT-2 text eval sets into model/eval/gpt2_eval/ (used by the accuracy tools --
# prover_exact_mimic.py, gpt2_argmax_accuracy.py, gpt2_ptq_baselines.py -- NOT by any ZK proof).
#   python3 tools/fetch_eval_data.py [wikitext|lambada|all]
import os, sys, json
os.makedirs("model/eval/gpt2_eval", exist_ok=True)
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("this needs the HuggingFace 'datasets' library:  pip install datasets")

what = sys.argv[1] if len(sys.argv) > 1 else "all"

if what in ("wikitext", "all"):
    ds = load_dataset("wikitext", "wikitext-2-v1", split="test")     # the <unk>-tokenised variant the file uses
    with open("model/eval/gpt2_eval/wikitext2_test.txt", "w") as f:
        f.write("".join(ds["text"]))
    print("wrote model/eval/gpt2_eval/wikitext2_test.txt")

if what in ("lambada", "all"):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    with open("model/eval/gpt2_eval/lambada_test.jsonl", "w") as f:
        for ex in ds:
            f.write(json.dumps({"text": ex["text"]}) + "\n")
    print("wrote model/eval/gpt2_eval/lambada_test.jsonl")
