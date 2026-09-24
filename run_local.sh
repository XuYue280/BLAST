#!/usr/bin/env bash
# BLAST on a single GPU, no Slurm -- same interface as the other baselines'
# run_local.sh so one driver script can sweep all of them.
#
#   ./run_local.sh setup                       pinned venv
#   ./run_local.sh doctor                      GPU + venv + import check
#   ./run_local.sh run facebook/opt-125m 0.60  rho = fraction of params KEPT
#   ./run_local.sh sweep                       both ratios x both models
#   ./run_local.sh collect                     -> local/runs/blast/results.csv
#
# Env:
#   VENV=<dir>        default ./.venv-blast   (keep it OFF a path with spaces)
#   PYBIN=<python>    interpreter used to BUILD the venv
#   RUNS_DIR=<dir>    default ./local/runs/blast
#   LOG_DIR=<dir>     default ./local/logs
#   GPUS=<list>       default 0
#   BLAST_NUM_BLOCKS  default 16   (upstream's default; "block size 16x16")
#   BLAST_NUM_ITER    default 300  (upstream's default)
#   BLAST_DELTA       default 0.1  (upstream's default)
#   EXTRA_ARGS        appended to the python call (e.g. --dense)
#
# rho IS the fraction KEPT. Upstream's --comp_ratio is the same quantity: its
# ranks table and scripts/decompose_llama.sh agree at comp_ratio=0.5 via
# comp_ratio = r*(M+N+nb^2)/(M*N). No 1-rho inversion anywhere.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=blast
VENV="${VENV:-$HERE/.venv-blast}"
RUNS_DIR="${RUNS_DIR:-$HERE/local/runs/$METHOD}"
LOG_DIR="${LOG_DIR:-$HERE/local/logs}"
SEED="${SEED:-42}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.0}"
TRANSFORMERS_SPEC="transformers==4.45.2"
NUMPY_SPEC="numpy==1.26.4"

log() { printf '[run_local] %s\n' "$*"; }
die() { printf '[run_local] ERROR: %s\n' "$*" >&2; exit 1; }
rho_tag() { awk -v r="$1" 'BEGIN{printf "%.0f", r*100}'; }

setup_env() {
  mkdir -p "$RUNS_DIR" "$LOG_DIR"
  export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
}

do_setup() {
  local force=0; [[ "${1:-}" == "--force" ]] && force=1
  local py="${PYBIN:-}"
  [[ -z "$py" ]] && { py=$(command -v python3.12 || command -v python3.11 || command -v python3) || die "no python3"; }
  if [[ -d "$VENV" && $force -eq 0 ]]; then
    log "$VENV exists -- reusing it (./run_local.sh setup --force to rebuild)"
  else
    (( force )) && rm -rf "$VENV"
    log "building $VENV with $py ($("$py" -V 2>&1))"
    "$py" -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel
  # einops is BLAST's own dependency (ops/blast_ops.py imports it); the rest
  # matches the other baselines so shared_eval.py behaves identically.
  "$VENV/bin/python" -m pip install -q "$TORCH_SPEC" "$NUMPY_SPEC" "$TRANSFORMERS_SPEC" \
      einops scipy safetensors sentencepiece datasets accelerate pyarrow pandas tqdm protobuf
  "$VENV/bin/python" - <<'PY'
import torch, transformers, numpy, einops
print(f"  transformers {transformers.__version__}  torch {torch.__version__} "
      f"(cuda {torch.version.cuda})  numpy {numpy.__version__}  einops {einops.__version__}  "
      f"cuda_available={torch.cuda.is_available()}")
assert transformers.__version__.startswith("4.45"), \
    "shared_eval.py's recipes were validated on transformers 4.45.x"
PY
  log "setup complete"
}

do_doctor() {
  setup_env
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  log "GPU:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
  "$VENV/bin/python" - <<'PY'
import sys, torch
sys.path.insert(0, "local"); sys.path.insert(0, ".")
import shared_eval  # noqa
from ops.blast_ops import blast_precond_gd, get_matrix  # noqa
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print("  imports OK (shared_eval + BLAST ops)")
PY
  log "doctor passed"
}

run_one() {
  local model="$1" rho="$2"
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  local short="${model##*/}"
  local tag="${short}_rho$(rho_tag "$rho")"
  local out="$RUNS_DIR/${tag}.json"
  local logf="$LOG_DIR/${METHOD}_${tag}.log"
  mkdir -p "$RUNS_DIR" "$LOG_DIR"
  echo "=========================================================="
  echo "[run_local] method=$METHOD model=$model rho=$rho"
  echo "[run_local] num_blocks=${BLAST_NUM_BLOCKS:-16} num_iter=${BLAST_NUM_ITER:-300}"
  echo "[run_local] out=$out"
  echo "[run_local] log=$logf"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "=========================================================="
  ( cd "$HERE" && "$VENV/bin/python" "$HERE/local/run_blast.py" \
      --model "$model" --rho "$rho" --seed "$SEED" \
      --num-blocks "${BLAST_NUM_BLOCKS:-16}" \
      --num-iter "${BLAST_NUM_ITER:-300}" \
      --delta "${BLAST_DELTA:-0.1}" \
      --out "$out" ${EXTRA_ARGS:-} 2>&1 | tee "$logf" )
  return "${PIPESTATUS[0]}"
}

do_sweep() {
  setup_env
  local models=("${@}")
  [[ ${#models[@]} -eq 0 ]] && models=(facebook/opt-125m meta-llama/Llama-3.2-1B)
  local fail=0
  for m in "${models[@]}"; do
    for r in 0.60 0.33; do
      if run_one "$m" "$r"; then :; else fail=$((fail+1)); log "FAIL $m rho=$r -- continuing"; fi
    done
  done
  log "sweep done, $fail failure(s)"
  return 0
}

do_collect() {
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  "$VENV/bin/python" - "$RUNS_DIR" <<'PY'
import csv, glob, json, os, re, sys
root = sys.argv[1]
CANON = re.compile(r"^[A-Za-z0-9._-]+_rho\d+\.json$")
rows, skipped = [], []
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    if os.path.basename(f) == "dense_eval_cache.json":
        continue
    if not CANON.match(os.path.basename(f)):
        skipped.append(f); continue
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    for corpus, ppl in (d.get("ppl") or {}).items():
        rows.append({
            "method": d.get("method"), "model": d.get("model"),
            "rho": d.get("rho_target"), "dense": d.get("dense"),
            "corpus": corpus, "ppl": ppl,
            "ppl_tokens": (d.get("ppl_tokens") or {}).get(corpus),
            "num_blocks": d.get("num_blocks"), "num_iter": d.get("num_iter"),
            "target_matrices": d.get("target_matrices"),
            "target_achieved_compression_ratio": d.get("target_achieved_compression_ratio"),
            "realised_linear_ratio": d.get("realised_linear_ratio"),
            "realised_total_ratio": d.get("realised_total_ratio"),
            "compress_seconds": d.get("compress_seconds"),
            "baseline": d.get("baseline"), "dtype": d.get("dtype"),
            "seed": d.get("seed"), "gpu": d.get("gpu"),
            "requested_compression_ratio": d.get("requested_compression_ratio"),
            "dense_ppl": (d.get("dense_ppl") or {}).get(corpus),
            "dense_ppl_tokens": (d.get("dense_ppl_tokens") or {}).get(corpus),
            "ppl_increase": (d.get("ppl_increase") or {}).get(corpus),
            "ppl_increase_percent": (d.get("ppl_increase_percent") or {}).get(corpus),
            "peak_cuda_allocated_mb": d.get("peak_cuda_allocated_mb"),
            "peak_cuda_reserved_mb": d.get("peak_cuda_reserved_mb"),
            "peak_cpu_rss_mb": d.get("peak_cpu_rss_mb"),
            "peak_cpu_rss_increase_mb": d.get("peak_cpu_rss_increase_mb"),
            "model_load_seconds": d.get("model_load_seconds"),
            "reconstruction_seconds": d.get("reconstruction_seconds"),
            "artifact_load_seconds": d.get("artifact_load_seconds"),
            "inference_seconds": d.get("inference_seconds"),
            "metric_seconds": d.get("metric_seconds"),
            "dense_inference_seconds": d.get("dense_inference_seconds"),
            "total_seconds": d.get("total_seconds"),
            "artifact_storage_mb": d.get("artifact_storage_mb"),
            "static_parameter_storage_mb": d.get("static_parameter_storage_mb"),
            "total_model_storage_mb": d.get("total_model_storage_mb"),
            "artifact_bits_per_compressed_parameter": d.get("artifact_bits_per_compressed_parameter"),
            "total_model_bits_per_parameter": d.get("total_model_bits_per_parameter"),
            "layers_evaluated": d.get("layers_evaluated"),
            "matrices_evaluated": d.get("matrices_evaluated"),
            "package_versions": json.dumps(d.get("package_versions") or {}),
            "last_updated": d.get("last_updated"),
            "source": f,
        })
if not rows:
    print("no canonical result files"); sys.exit(0)
rows.sort(key=lambda r: (r["model"] or "", str(r["rho"]), r["corpus"]))
out = os.path.join(root, "results.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    cols = list(rows[0])
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    w = csv.DictWriter(fh, fieldnames=cols, restval="")
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} row(s) -> {out}")
for f in skipped:
    print(f"  skipped non-canonical: {f}")
print(f"{'model':<26}{'rho':>6}{'corpus':>12}{'ppl':>12}")
for r in rows:
    print(f"{(r['model'] or '').split('/')[-1]:<26}{str(r['rho'] or 'dense'):>6}"
          f"{r['corpus']:>12}{r['ppl']:>12.4f}")
PY
}

case "${1:-}" in
  setup)   shift; do_setup "$@" ;;
  doctor)  do_doctor ;;
  run)     [[ $# -eq 3 ]] || die "usage: ./run_local.sh run MODEL RHO"; setup_env; run_one "$2" "$3" ;;
  sweep)   shift; do_sweep "$@" ;;
  collect) do_collect ;;
  help|-h|--help) sed -n '2,30p' "$0" ;;
  *) die "unknown command: ${1:-}. Try: setup | doctor | run | sweep | collect | help" ;;
esac
