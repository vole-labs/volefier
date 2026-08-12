#!/usr/bin/env bash
# Fetch / build ONE model the ZK tests need -- does NOT pull everything at once; pass a target.
#   tools/fetch_models.sh <target>
# See model/README.md for what each model is.  Run with no target (or 'list') for the menu.
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root
export KMP_DUPLICATE_LIB_OK=TRUE             # torch+onnxruntime both link libomp; avoid double-init abort
PY=${PYTHON:-python3}
ZOO=https://github.com/onnx/models/raw/main/validated/vision/classification

usage() {
  cat <<'EOF'
Usage: tools/fetch_models.sh <target>          (one target per call)

MODELS (needed by the ZK proofs):
  gpt2         model/gpt2-12_nocache/           HF 'gpt2' (12-layer)  -> GPT-2 tests' default + quarot source
  gpt2-quarot  model/gpt2-12-quarot.onnx        derive from gpt2     -> ml-quant GPT-2   [needs: gpt2]
  bert         model/bert_base_fp32_basic.onnx + model/bert_base_qdq.onnx  -> ml-fixed / ml-quant BERT
  resnet18     model/resnet18-v1-7-qdq.onnx     ONNX Model Zoo fp32 + quantize (fp32 -> eval/)
  resnet50     model/resnet50-v1-7-qdq.onnx     ONNX Model Zoo fp32 + quantize (fp32 -> eval/)
  vgg16        model/vgg16-12-qdq.onnx          ONNX Model Zoo fp32 + quantize (fp32 -> eval/)

LLMs (fp32 -> ml-fixed, QDQ int8 -> ml-quant; QDQ calibrated on WikiText-2):
  qwen         model/qwen2.5-0.5b/model.onnx + -qdq.onnx   HF Qwen/Qwen2.5-0.5B-Instruct (OPEN)
  gemma        model/gemma-3-270m/model.onnx + -qdq.onnx   HF google/gemma-3-270m-it (GATED; DeepProve's model)
  llama        model/llama-3.2-1b/model.onnx + -qdq.onnx   HF meta-llama/Llama-3.2-1B (GATED)
               override the HF repo:  tools/fetch_models.sh <qwen|gemma|llama> <hf_repo_id>
               (run 'tools/fetch_models.sh wikitext' first so QDQ calibrates on real text)

EVAL DATA (for the accuracy tools only -- NOT needed by any proof):
  wikitext     model/eval/gpt2_eval/wikitext2_test.txt   WikiText-2 test  (HF datasets)
  lambada      model/eval/gpt2_eval/lambada_test.jsonl   LAMBADA test     (HF datasets)
  gpt2-eval    both of the above
  imagenet     model/eval/imagenet/             ILSVRC2012 val (manual -- registration)

Examples:  tools/fetch_models.sh gpt2      tools/fetch_models.sh bert      tools/fetch_models.sh gpt2-eval
EOF
}

# build one LLM: fp32 ONNX (ml-fixed) + WikiText-2-calibrated QDQ int8 (ml-quant)
#   $1=out-dir-name  $2=default HF repo  $3=optional HF repo override
llm_build() {
  local out="$1" hf="${3:-$2}"
  [ -f model/eval/gpt2_eval/wikitext2_test.txt ] || \
    echo ">> note: run 'tools/fetch_models.sh wikitext' for real QDQ calibration (else random fallback)"
  $PY tools/export_llm_onnx.py "$hf" "model/$out"
  $PY tools/quantize_llm_qdq.py "model/$out/model.onnx" "model/$out-qdq.onnx" "$hf"
}

case "${1:-list}" in
  list|-h|--help|help) usage ;;

  gpt2)         $PY tools/export_gpt2_nocache.py gpt2 model/gpt2-12_nocache ;;
  gpt2-quarot)
    [ -f model/gpt2-12_nocache/model.onnx ] || { echo "need gpt2 first:  tools/fetch_models.sh gpt2"; exit 1; }
    PYTHONPATH=. $PY tools/export_quarot_onnx.py model/gpt2-12_nocache/model.onnx model/gpt2-12-quarot.onnx ;;

  bert)         $PY tools/export_bert.py ;;

  resnet18)     $PY tools/quantize_cnn.py "$ZOO/resnet/model/resnet18-v1-7.onnx"  model/resnet18-v1-7-qdq.onnx  model/eval/resnet18-v1-7/resnet18-v1-7.onnx ;;
  resnet50)     $PY tools/quantize_cnn.py "$ZOO/resnet/model/resnet50-v1-7.onnx"  model/resnet50-v1-7-qdq.onnx  model/eval/resnet50-v1-7/resnet50-v1-7.onnx ;;
  vgg16)        $PY tools/quantize_cnn.py "$ZOO/vgg/model/vgg16-7.onnx"           model/vgg16-12-qdq.onnx       model/eval/vgg16/vgg16.onnx ;;

  qwen)         llm_build qwen2.5-0.5b  Qwen/Qwen2.5-0.5B-Instruct "${2:-}" ;;
  gemma)        llm_build gemma-3-270m  google/gemma-3-270m-it     "${2:-}" ;;
  llama)        llm_build llama-3.2-1b  meta-llama/Llama-3.2-1B    "${2:-}" ;;

  wikitext)     $PY tools/fetch_eval_data.py wikitext ;;
  lambada)      $PY tools/fetch_eval_data.py lambada ;;
  gpt2-eval)    $PY tools/fetch_eval_data.py all ;;

  imagenet)
    cat <<'EOF'
ImageNet val (6.3 GB, needed ONLY by tools/accuracy_baselines.py -- not by any ZK proof):
  1. Download the ILSVRC2012 validation set from https://image-net.org (registration required).
  2. Arrange as:  model/eval/imagenet/val/<wnid>/*.JPEG   and   model/eval/imagenet/class_map.csv
EOF
    ;;

  *) echo "unknown target '$1'"; echo; usage; exit 1 ;;
esac
