#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
ROOT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_flatpose/gates/batch_benchmark_20260723
mkdir -p "$ROOT"
: > "$ROOT/summary.tsv"
printf 'batch	exit	batch_seconds	samples_per_second	allocated_mb	reserved_mb	max_sequence_tokens
' >> "$ROOT/summary.tsv"
for BS in 8 16 24 32; do
  OUT="$ROOT/bs${BS}"
  if [[ -e "$OUT" ]]; then
    echo "Refusing existing $OUT" >&2
    exit 3
  fi
  set +e
  /root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py     --output-dir "$OUT" --run-kind smoke --max-samples 256 --epochs 1     --max-optimizer-steps 1 --batch-size "$BS" --grad-accum-steps 1 --num-workers 4     --horizontal-flip-prob 0 --max-people 25 --person-query-layers 3     --pose-decoder-layers 2 --scheduler-type constant --warmup-ratio 0     --visual-lr 5e-6 --language-lr 5e-6 --token-lr 1e-5 --head-lr 1e-4     --grad-clip 1 --skip-save --log-every 1 > "$ROOT/bs${BS}.log" 2>&1
  CODE=$?
  set -e
  /root/miniconda3/envs/human4b/bin/python - "$BS" "$CODE" "$OUT/events.jsonl" >> "$ROOT/summary.tsv" <<'PY2'
import json,sys
bs,code,path=sys.argv[1:]
try:
 rows=[json.loads(x) for x in open(path) if '"event": "train"' in x]
 row=rows[-1]
 vals=[row.get(k,'') for k in ['batch_seconds','samples_per_second','cuda_allocated_mb','cuda_reserved_mb','max_sequence_tokens']]
except Exception:
 vals=['','','','','']
print('	'.join(map(str,[bs,code,*vals])))
PY2
  cat "$ROOT/summary.tsv"
  if [[ "$CODE" -ne 0 ]]; then
    tail -n 40 "$ROOT/bs${BS}.log"
    break
  fi
done
