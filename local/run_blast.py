#!/usr/bin/env python3
"""BLAST at a given parameter-reduction ratio, evaluated through shared_eval.

Localisation notes -- what was adapted from upstream and why:

1. UPSTREAM HARD-CODES THE RANKS AND *DERIVES* THE RATIO.
   blastify_llama.py has `ranks = {"q_proj": 1024, ...}` and then computes
   `comp_ratio = r*(M+N+num_blocks**2)/M/N`; its own `--comp_ratio` flag is never
   read except to name the output directory. We need the inverse: given rho, solve
   for r. Verified against upstream's own numbers -- r=1024 on llama-7b's
   4096x4096 with num_blocks=4 gives 0.5010, and scripts/decompose_llama.sh passes
   `--comp_ratio 0.5`. So:

       comp_ratio == fraction of parameters KEPT == our rho.     (no 1-rho flip)
       r = rho * M * N / (M + N + num_blocks**2)

   A BLAST factor stores B (b1,p,r), C (b2,r,q), D (r,b1,b2) with b1*p = M and
   b2*q = N, i.e. r*(M+N+nb^2) parameters -- which is where that formula comes from.

2. TARGET MATRICES ONLY; EVERYTHING ELSE STAYS DENSE.
   Upstream targets q/k/v/o_proj and gate/up/down_proj by substring, which is
   Llama-specific. OPT names them q/k/v/out_proj and fc1/fc2. Both sets are
   decoder linears only -- lm_head and the embeddings are never touched, matching
   the other baselines. Counts land on 72 target matrices for opt-125m and 112 for
   Llama-3.2-1B, the same numbers the SVD baselines and the reference bench report.

3. DECOMPOSE -> RECONSTRUCT -> EVALUATE.
   Upstream saves B/C/D to disk and evaluates through its own BLAST modules, which
   exist only for Llama. `ops.blast_ops.get_matrix(B,C,D)` rebuilds the dense
   matrix the factorisation represents, so we write that back into the original
   nn.Linear and evaluate with the same shared_eval.py as every other baseline.
   The perplexity is identical to what BLAST's own kernels would produce -- the
   factorisation is unchanged, only the inference path differs -- and it is the
   only way to get one comparable number across four baselines and two model
   families. Parameter accounting still uses the FACTORISED count, not the dense
   reconstruction, so the ratios mean what they say.
"""
import argparse, json, os, sys, time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from shared_eval import (evaluate_all, count_parameters, arks_fields,
                         PeakMemory, load_dense_cache, save_dense_cache)

SEQLEN = 2048

# Decoder linears only, per family. lm_head / embeddings are never targeted.
TARGETS = {
    "opt":   ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}


def family_of(model):
    mt = (getattr(model.config, "model_type", "") or "").lower()
    if "opt" in mt:
        return "opt"
    if "llama" in mt:
        return "llama"
    raise SystemExit(f"unsupported model_type {mt!r} -- add its decoder linear names to TARGETS")


def rank_for(M, N, rho, num_blocks):
    """Invert upstream's comp_ratio formula. See note 1 in the module docstring."""
    r = int(round(rho * M * N / (M + N + num_blocks ** 2)))
    return max(r, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, required=True, help="fraction of parameters KEPT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--num-blocks", type=int, default=16, help="BLAST block count (upstream default)")
    ap.add_argument("--num-iter", type=int, default=300, help="GD iterations (upstream default)")
    ap.add_argument("--delta", type=float, default=0.1, help="upstream default")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(a.seed)

    _t_wall0 = time.perf_counter()
    _mem = PeakMemory(); _mem.__enter__()

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    _t_load = time.perf_counter() - _t_wall0

    before = count_parameters(model)
    fam = family_of(model)
    tags = TARGETS[fam]

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "lm_head" not in n
               and n.split(".")[-1] in tags]
    print(f"[blast] family={fam}  target matrices={len(targets)}  "
          f"num_blocks={a.num_blocks}  num_iter={a.num_iter}", flush=True)

    compressed_target_params = 0
    dense_target_params = 0
    per_matrix = []
    t0 = time.perf_counter()

    if not a.dense:
        from ops.blast_ops import blast_precond_gd, get_matrix
        dev = torch.device("cuda")
        parent = {}
        for pn, pm in model.named_modules():
            for cn, cm in pm.named_children():
                parent[id(cm)] = (pm, cn)

        for idx, (name, mod) in enumerate(targets, 1):
            M, N = mod.out_features, mod.in_features
            if M % a.num_blocks or N % a.num_blocks:
                raise SystemExit(f"{name}: {M}x{N} not divisible by num_blocks={a.num_blocks}")
            r = rank_for(M, N, a.rho, a.num_blocks)
            W = mod.weight.data.float()
            ts = time.perf_counter()
            B, C, D = blast_precond_gd(W, num_blocks=a.num_blocks, r=r, T=a.num_iter,
                                       device=dev, delta=a.delta, end_factor=0.0,
                                       verbose=False)
            B, C, D = B.cpu().float(), C.cpu().float(), D.cpu().float()
            Wr = get_matrix(B, C, D)
            err = (torch.norm(Wr - W) / torch.norm(W)).item()
            mod.weight.data = Wr.to(mod.weight.dtype)
            n_fact = B.numel() + C.numel() + D.numel()
            compressed_target_params += n_fact
            dense_target_params += M * N
            per_matrix.append({
                "name": name, "shape": [M, N], "rank": r,
                "dense_parameter_count": M * N,
                "compressed_parameter_count": int(n_fact),
                "requested_compression_ratio": a.rho,
                "achieved_matrix_ratio": n_fact / (M * N),
                "relative_frobenius_error": err,
                "decomposition_time_seconds": time.perf_counter() - ts,
            })
            del B, C, D, Wr, W
            if idx % 10 == 0 or idx == len(targets):
                el = time.perf_counter() - t0
                print(f"[blast] {idx}/{len(targets)}  {el:.0f}s elapsed, "
                      f"~{el/idx*(len(targets)-idx):.0f}s left", flush=True)
            torch.cuda.empty_cache()
    else:
        dense_target_params = sum(m.out_features * m.in_features for _, m in targets)
        compressed_target_params = dense_target_params

    compress_s = time.perf_counter() - t0
    _n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    _n_matrices = len(targets)

    model = model.half().to("cuda")
    _t_eval0 = time.perf_counter()
    res = evaluate_all(model, tok, device="cuda")
    _t_eval1 = time.perf_counter()

    # Parameter accounting uses the FACTORISED count, not the dense reconstruction
    # that was written back into the module (see note 3).
    after = dict(before)
    delta = dense_target_params - compressed_target_params
    after = {"total_params": before["total_params"] - delta,
             "linear_params": before["linear_params"] - delta}

    realised_linear = after["linear_params"] / before["linear_params"]
    realised_total = after["total_params"] / before["total_params"]
    achieved = compressed_target_params / max(dense_target_params, 1)

    payload = {
        "method": "blast", "model": a.model,
        "rho_target": None if a.dense else a.rho,
        "dense": a.dense, "seed": a.seed,
        "num_blocks": a.num_blocks, "num_iter": a.num_iter, "delta": a.delta,
        "target_matrices": _n_matrices,
        "target_achieved_compression_ratio": achieved,
        "params_before": before, "params_after": after,
        "realised_total_ratio": realised_total,
        "realised_linear_ratio": realised_linear,
        "compress_seconds": compress_s,
        "ppl": {c: v["ppl"] for c, v in res.items()},
        "ppl_tokens": {c: v["ppl_tokens"] for c, v in res.items()},
        "per_matrix": per_matrix,
    }

    _runs_root = os.path.dirname(a.out)
    if a.dense:
        save_dense_cache(_runs_root, res, inference_seconds=_t_eval1 - _t_eval0)
    _dense_res = load_dense_cache(_runs_root)
    payload.update(arks_fields(
        method="blast", model=a.model, rho=(None if a.dense else a.rho),
        res=res, params_before=before, params_after=after,
        dense=a.dense, seed=a.seed, dense_res=_dense_res,
        timings={
            "inference_seconds": _t_eval1 - _t_eval0,
            "metric_seconds": _t_eval1 - _t_eval0,
            "model_load_seconds": _t_load,
            "reconstruction_seconds": compress_s if not a.dense else 0.0,
            "artifact_load_seconds": 0.0,
            "dense_inference_seconds": (_dense_res.get("_meta") or {}).get("dense_inference_seconds"),
            "total_seconds": time.perf_counter() - _t_wall0,
        },
        peak_memory=_mem.fields(),
        layers_evaluated=_n_layers, matrices_evaluated=_n_matrices))
    _mem.__exit__()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\n" + json.dumps(payload["ppl"], indent=2))
    print(f"[check] target matrices={_n_matrices}  achieved ratio={achieved:.4f}")
    print(f"realised linear ratio = {realised_linear:.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
