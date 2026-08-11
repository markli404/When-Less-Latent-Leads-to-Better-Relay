import time



import os
import socket
import numpy as np
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
from typing import Dict, List, Tuple
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from compression_methods import *
from utils import auto_device, set_seed

METRIC_NAMES = [
    "r_perp",
    "recovery_R",
    "recovery_cos",
    "pca_evr",
    "ad_over_as",
    "inj_norm",
]

WANDB_FILE_SAVED = False


def build_wandb_run_name(base_name):
    job_id = os.getenv("SLURM_JOB_ID")
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")

    if job_id and array_task_id:
        return f"{base_name}_job{job_id}_task{array_task_id}"
    if job_id:
        return f"{base_name}_job{job_id}"
    return base_name


def build_wandb_run_dir(output_folder):
    run_token = f"{socket.gethostname()}-{os.getpid()}-{int(time.time() * 1000)}"
    return os.path.join(output_folder, "wandb", run_token)


def init_metric_store(metric_names=None):
    metric_names = metric_names or METRIC_NAMES
    return {
        "sample_ids": [],
        "correct": [],
        **{
            f"{name}_{i}": []
            for name in metric_names
            for i in range(3)
        },
    }


def append_metrics(metric_store, res, problem_idx, metric_names=None):
    metric_names = metric_names or METRIC_NAMES

    metric_store["sample_ids"].append(problem_idx)
    metric_store["correct"].append(int(bool(res.get("correct"))))

    for name in metric_names:
        for i in range(3):
            key = f"{name}_{i}"
            x = res.get(key)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            elif x is not None:
                x = np.asarray(x)
            metric_store[key].append(x)


def _stack_optional_metric(values, key):
    template = None
    for value in values:
        if value is None:
            continue
        template = np.asarray(value, dtype=np.float32)
        break

    if template is None:
        return np.empty((len(values), 0), dtype=np.float32)

    stacked = []
    for value in values:
        if value is None:
            stacked.append(np.full(template.shape, np.nan, dtype=np.float32))
            continue

        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != template.shape:
            raise ValueError(
                f"Inconsistent metric shape for {key}: expected {template.shape}, got {arr.shape}"
            )
        stacked.append(arr)

    return np.stack(stacked, axis=0)


def save_metric_store(metric_store, output_path, use_wandb=False):
    save_dict = {
        "sample_ids": np.array(metric_store["sample_ids"], dtype=np.int64),
        "correct": np.array(metric_store["correct"], dtype=np.int64),
    }

    for name in METRIC_NAMES:
        for i in range(3):
            key = f"{name}_{i}"
            save_dict[key] = _stack_optional_metric(metric_store[key], key)

    np.savez_compressed(output_path, **save_dict)

    if use_wandb:
        wandb.save(output_path, policy="live")

def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

def process_batch(
        method,
        batch,
        processed,
        preds,
        progress,
        max_samples,
        args,
        metric_store,
        output_file,
):
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds

    current_batch = batch[:remaining]
    results = method.run_batch(current_batch)

    if len(results) > remaining:
        results = results[:remaining]

    batch_start = processed

    output_folder = getattr(args, "output_folder", "results")
    os.makedirs(output_folder, exist_ok=True)



    use_wandb = getattr(args, "use_wandb", False)
    verbose = getattr(args, "verbose", False)

    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1

        # collect metric into global store
        append_metrics(metric_store, res, problem_idx)

        agents = res.get("agents", [])
        agent_traces = []
        for a in agents:
            trace = {
                "name": a.get("name", "Agent"),
                "role": a.get("role", ""),
                "input": a.get("input", "").rstrip(),
                "input_ids": a.get("input_ids", []),
                "output": a.get("output", "").rstrip(),
                "latent_steps": a.get("latent_steps", None),
            }
            for optional_key in [
                "selected_prompt_positions_matrix",
            ]:
                if optional_key in a:
                    trace[optional_key] = a.get(optional_key)
            agent_traces.append(trace)

        log_record = {
            "id": problem_idx,
            "question": res.get("question", "").strip(),
            "prediction": res.get("prediction"),
            "gold": res.get("gold"),
            "correct": res.get("correct"),
            "agent_traces": agent_traces,
        }

        if verbose:
            print(f"\n==================== Problem #{problem_idx} ====================")
            print(f"Question: {log_record['question']}")
            for trace in agent_traces:
                print(f"----- Agent: {trace['name']} ({trace['role']}) -----")
                print("[To Tokenize]")
                print(trace["input"])
                if trace["latent_steps"] is not None:
                    print("[Latent Steps]")
                    print(trace["latent_steps"])
                print("[Output]")
                print(trace["output"])
                print("----------------------------------------------")
            print(
                f"Result: Pred={log_record['prediction']} | "
                f"Gold={log_record['gold']} | OK={log_record['correct']}"
            )

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))

    return processed, preds


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for this LatentMAS research fork."
    )

    # Experiment selection
    parser.add_argument(
        "--method",
        choices=["baseline", "text_mas", "latent_mas"],
        required=True,
        help="Experiment method. In this fork, only 'latent_mas' is maintained.",
    )
    parser.add_argument("--model_name", type=str, required=True, choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"])
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to evaluate (-1 uses the full split).")
    parser.add_argument("--shard_index", type=int, default=0, help="Data-parallel shard index for this process (0-based). Use with --num_shards to split the eval set across multiple GPUs.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards. Each process handles samples where (global_index %% num_shards) == shard_index. Set >1 for multi-GPU data-parallel bs=1 eval.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Prompt style for MAS methods.")

    # Runtime and generation
    parser.add_argument("--device", type=str, default="cuda", help="Device string passed to torch (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--split", type=str, default="test", help="Dataset split when the loader supports custom splits.")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max generated tokens for baseline/text_mas and the latent_mas judger.")
    parser.add_argument("--latent_steps", type=int, default=10, help="Number of latent rollout steps for each non-judger agent in latent_mas.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for text decoding. Use 0 for greedy decoding.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling threshold for text decoding.")
    parser.add_argument("--generate_bs", type=int, default=4, help="Batch size for method.run_batch().")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context truncation length in prompts.py (-1 usually means no truncation).")
    parser.add_argument("--think", action="store_true", help="Append '<think>' to agent prompts in latent_mas.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print per-sample agent traces to stdout.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.", default=False)
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="Optional Weights & Biases project name. Defaults to LatentMAS_{seed}.",
    )

    # OBF-related args
    parser.add_argument("--pca_rank", type=int, default=8, help="PCA rank used by LOBF/HOBF-style compressors.")
    parser.add_argument("--kv_budget", type=int, default=32, help="Prompt KV budget for compression-based methods.")
    parser.add_argument("--sink_size", type=int, default=4, help="Prompt sink tokens kept outside the main budget.")
    parser.add_argument("--inject_mode", type=str, default="uniform", choices=["uniform", "attn"],
                        help="OBF residual injection: 'uniform' adds the same delta to every kept value; "
                             "'attn' distributes it by kept-token attention weights. Used by lobf/hobf/lobf_evr.")
    parser.add_argument("--quant_bits", type=int, default=8, choices=[2, 4, 8],
                        help="Relay payload precision for full_quant / lobf_quant (fake quantization).")

    # Output and compression
    parser.add_argument("--output_folder", type=str, default="results", help="Path to save logs")
    parser.add_argument("--compressor", type=str, default='full', help="KV compressor for latent_mas (e.g., full, gonly, headwise, layerwise, lobf, lobf_evr, lobf_naive, lobf_no_proj, lobf_no_scale, lobf_max_p, hobf, hobf_naive, hobf_no_proj, hobf_no_scale, hobf_max_p). For lobf_evr, --pca_rank is the EVR threshold in percent (90 -> tau=0.90).")
    parser.add_argument("--latent_space_realign", action="store_true", help="Apply output->input embedding space realignment to latent hidden states before feeding back as inputs_embeds. Default: OFF (matches upstream Gen-Verse/LatentMAS). Pass the flag to enable.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)

    model_name = args.model_name.split('/')[-1]
    name = f"{args.task}_{model_name}_{args.compressor}_{args.latent_steps}_bs{args.generate_bs}"

    if args.method != "latent_mas":
        raise NotImplementedError(
            f"Method '{args.method}' is not maintained in this fork. "
            "Use '--method latent_mas'."
        )

    # ---------------------------------------------------------------------
    # Initialize wandb FIRST so we can use its run.id as the primary run key.
    # When wandb is disabled, fall back to a locally generated unique id
    # (params hash + timestamp + pid). Either way, the run gets a dedicated
    # isolated output directory — two concurrent jobs can never collide.
    # ---------------------------------------------------------------------
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run without --use_wandb.")
        wandb_name = build_wandb_run_name(name)
        wandb_dir = build_wandb_run_dir(args.output_folder)
        wandb_project = args.project_name or f"LatentMAS_{args.seed}"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            dir=wandb_dir,
            config=vars(args),
            reinit=True,
        )
        # wandb.run.id is an 8-char globally-unique id generated per run.
        run_uid = wandb.run.id
    else:
        import hashlib
        _param_blob = (
            f"{args.task}|{model_name}|{args.method}|{args.compressor}"
            f"|{args.latent_steps}|{args.generate_bs}|{args.seed}"
            f"|{args.prompt}|{args.max_new_tokens}"
            f"|{getattr(args, 'pca_rank', '')}|{getattr(args, 'latent_space_realign', False)}"
        )
        _params_hash = hashlib.sha1(_param_blob.encode()).hexdigest()[:8]
        _timestamp = time.strftime("%Y%m%d-%H%M%S")
        _pid = os.getpid()
        run_uid = f"{_params_hash}_{_timestamp}_p{_pid}"

    # Use run_uid directly as the dir name. All hyperparams are already tracked
    # in wandb (or in config.json for non-wandb runs), so no need to encode them
    # in the filesystem path.
    run_id = run_uid
    run_dir = os.path.join(args.output_folder, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # setup output files names (isolated inside run_dir)
    output_file = os.path.join(run_dir, "raw_output.txt")
    obf_output_file = os.path.join(run_dir, "obf_metric.npz")

    # Fresh run_dir is always empty, but keep race-safe semantics in case a
    # future refactor repoints these paths elsewhere.
    for file_path in [output_file, obf_output_file]:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    # Persist the exact config for reproducibility
    try:
        with open(os.path.join(run_dir, "config.json"), "w") as _cf:
            json.dump(
                {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                 for k, v in vars(args).items()},
                _cf, indent=2,
            )
    except Exception:
        pass  # config dump is best-effort, never block the run

    if args.use_wandb:
        # Touch the output file so wandb.save has something to track
        with open(output_file, "a", encoding="utf-8") as f:
            pass
        wandb.save(output_file, policy="live")

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.compressor == "full":
        compressor = Full(sink_size=args.sink_size, kv_budget=args.kv_budget)
    elif args.compressor == "full_quant":
        # Full relay fake-quantized to --quant_bits (KV-quantization baseline).
        compressor = FullQuant(sink_size=args.sink_size, kv_budget=args.kv_budget,
                               quant_bits=args.quant_bits)
    elif args.compressor == "lmerge":
        # Layerwise selection + one-by-one key-similarity token merging baseline.
        compressor = LMerge(sink_size=args.sink_size, kv_budget=args.kv_budget)
    elif args.compressor == "lobf_quant":
        # OBF + --quant_bits fake-quant of the compressed relay (composition demo).
        compressor = LOBFQuant(sink_size=args.sink_size, kv_budget=args.kv_budget,
                               pca_rank=args.pca_rank, inject_mode=args.inject_mode,
                               quant_bits=args.quant_bits)
    elif args.compressor == "gonly":
        compressor = GenerationOnly(sink_size=args.sink_size, kv_budget=args.kv_budget)
    elif args.compressor == "headwise":
        compressor = Headwise(sink_size=args.sink_size, kv_budget=args.kv_budget)
    elif args.compressor == "layerwise":
        compressor = Layerwise(sink_size=args.sink_size, kv_budget=args.kv_budget)
    elif args.compressor == "lobf":
        compressor = LOBF(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                          inject_mode=args.inject_mode)
    elif args.compressor == "lobf_evr":
        # EVR-adaptive rank: --pca_rank carries the EVR threshold as a percent
        # (e.g. --pca_rank 90 -> tau=0.90). See compression_methods/LOBFEvr.py.
        compressor = LOBFEvr(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                             inject_mode=args.inject_mode)
    elif args.compressor == "lobf_batched":
        # LOBF with the per-head SVD loop batched over KV heads. Identical math,
        # one set of batched torch.linalg calls instead of H_kv Python-level ones.
        compressor = LOBFBatched(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                                 inject_mode=args.inject_mode)
    elif args.compressor == "lobf_fast":
        # Orthogonal subspace iteration for the top-p residual directions: only
        # GEMMs and a small QR, no SVD. Measured 6.15x faster than the full-SVD
        # path end to end, with the injected delta agreeing to 6.6e-5 at the
        # median and 2.1e-4 at p99, and no layer of 432 beyond 1e-2.
        # Iteration count, oversampling, QR cadence, warm start and residual
        # batching are not CLI flags. They are numerical properties of the
        # factorization, not experiment settings, and every one of them has a
        # single measured right answer; edit LOBFFast to change one. Carrying
        # a second copy of the defaults in argparse is what silently overrode the
        # class twice, since an argparse default is a value, not an absence.
        compressor = LOBFFast(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                                  inject_mode=args.inject_mode)
    elif args.compressor == "hobf_fast":
        # Headwise selector, same subspace iteration as lobf_fast. Tuning
        # lives in HOBFFast, not on the command line.
        compressor = HOBFFast(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                                  inject_mode=args.inject_mode)
    elif args.compressor == "lobf_gram":
        # Exact top-p residual directions via the Gram matrix G = R^T R, whose
        # eigenvectors are R's right singular vectors, batched over KV heads.
        # Same subspace as the full SVD (~1e-6 relative), far less arithmetic.
        compressor = LOBFGram(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                              inject_mode=args.inject_mode)
    elif args.compressor == "lobf_per_token":
        # Per-token soft-assignment injection. See compression_methods/LOBFPerToken.py.
        compressor = LOBFPerToken(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "lobf_naive":
        compressor = LOBFNaive(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "lobf_no_proj":
        compressor = LOBFNoProj(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "lobf_no_scale":
        compressor = LOBFNoScale(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "lobf_max_p":
        compressor = LOBFMaxP(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "hobf":
        compressor = HOBF(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank,
                          inject_mode=args.inject_mode)
    elif args.compressor == "hobf_naive":
        compressor = HOBFNaive(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "hobf_no_proj":
        compressor = HOBFNoProj(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "hobf_no_scale":
        compressor = HOBFNoScale(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    elif args.compressor == "hobf_max_p":
        compressor = HOBFMaxP(sink_size=args.sink_size, kv_budget=args.kv_budget, pca_rank=args.pca_rank)
    else:
        raise ValueError(f"Unknown compressor {args.compressor}")

    method = LatentMASMethod(
        model,
        compressor=compressor,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        **common_kwargs,
        generate_bs=args.generate_bs,
        args=args,
    )


    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    

    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    # Materialize so we can shard (and honor --max_samples deterministically).
    dataset_iter = list(dataset_iter)
    total_dataset_size = len(dataset_iter)

    # Data-parallel sharding: keep only samples whose global index falls in
    # this process's shard. Uses modulo (stride) rather than contiguous slice
    # so each shard sees a roughly equal mix of easy/hard samples regardless
    # of dataset ordering.
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(
            f"--shard_index must be in [0, num_shards). Got shard_index={args.shard_index}, num_shards={args.num_shards}."
        )
    if args.num_shards > 1:
        dataset_iter = [
            item for i, item in enumerate(dataset_iter)
            if (i % args.num_shards) == args.shard_index
        ]
        print(
            f"[shard {args.shard_index}/{args.num_shards}] "
            f"{len(dataset_iter)} / {total_dataset_size} samples on this process."
        )

    if args.max_samples == -1:
        args.max_samples = len(dataset_iter)
    else:
        # --max_samples is interpreted as per-shard when sharded, matching the
        # usual "run N samples on this process" semantics.
        args.max_samples = min(args.max_samples, len(dataset_iter))

    metric_store = init_metric_store()

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break

        batch.append(item)

        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method=method,
                batch=batch,
                processed=processed,
                preds=preds,
                progress=progress,
                max_samples=args.max_samples,
                args=args,
                metric_store=metric_store,
                output_file=output_file,
            )
            batch = []

            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method=method,
            batch=batch,
            processed=processed,
            preds=preds,
            progress=progress,
            max_samples=args.max_samples,
            args=args,
            metric_store=metric_store,
            output_file=output_file,
        )

    save_metric_store(metric_store, obf_output_file, args.use_wandb)
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    sum_lat_inf_time = sum(item.get("latent_inference_time", 0) for item in preds)
    sum_txt_inf_time = sum(item.get("text_inference_time", 0) for item in preds)
    sum_comp_time = sum(item.get("compression_time", 0) for item in preds)
    sum_comp_core = sum(item.get("compression_core_time", 0) for item in preds)
    sum_comm_bits = sum(item.get("communication_overhead", 0) for item in preds)
    sum_prompt = sum(item.get("prompt_len", 0) for item in preds)
    sum_token = sum(item.get("token_usage", 0) for item in preds)
    sum_peak = sum(item.get("peak_overhead", 0) for item in preds)
    # PCA explained-variance ratio: pool across (samples × 3 rounds × heads/layers).
    # Each metric_store entry is a list of arrays; flatten + drop NaN, then summarise.
    pca_evr_values = []
    for round_idx in range(3):
        for arr in metric_store.get(f"pca_evr_{round_idx}", []):
            if arr is None:
                continue
            flat = np.asarray(arr, dtype=np.float64).ravel()
            pca_evr_values.append(flat[np.isfinite(flat)])
    pca_evr_pooled = np.concatenate(pca_evr_values) if pca_evr_values else np.array([])

    metrics = {
        "method": args.method,
        "compressor": args.compressor,
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time": round(total_time, 4),
        "time_per_sample_sec": round(total_time / args.max_samples, 4),

        "avg_latent_inference_time_s": round(sum_lat_inf_time / args.max_samples, 4),
        "avg_text_inference_time_s": round(sum_txt_inf_time / args.max_samples, 4),
        "avg_compression_time_s": round(sum_comp_time / args.max_samples, 4),
        # Same call with the diagnostic blocks excluded: what the operator
        # would cost without the paper's instrumentation.
        "avg_compression_core_s": round(sum_comp_core / args.max_samples, 4),
        "avg_communication_MB": round(sum_comm_bits / args.max_samples, 4),
        "prompt_len": round(sum_prompt / args.max_samples, 4),
        "token_usage": round(sum_token / args.max_samples, 4),
        "peak_overhead": round(sum_peak / args.max_samples, 4),

        "pca_evr_mean":   round(float(np.mean(pca_evr_pooled)),   4) if pca_evr_pooled.size else None,
        "pca_evr_median": round(float(np.median(pca_evr_pooled)), 4) if pca_evr_pooled.size else None,
    }
    print(json.dumps(metrics, ensure_ascii=False))

    if args.use_wandb and wandb is not None:
        wandb.log(metrics)
        wandb.finish()



if __name__ == "__main__":
    main()
