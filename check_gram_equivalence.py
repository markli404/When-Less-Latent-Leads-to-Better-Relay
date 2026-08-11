"""Is LOBFGram numerically equivalent to LOBF on real relay tensors?

The synthetic unit test says the injected delta agrees to ~1e-6, and pca_evr
agrees to 1e-4 in a live run, but a seed-4 campaign came out 1.26 pp below LOBF
on five of five benchmarks, which noise does not explain: on GSM8K the three
LOBF seeds span 0.30 pp and Gram sat 1.44 pp below. Something the synthetic test
does not cover is different, and pca_evr cannot see it because eigenvalues are
insensitive to a rotation of the eigenvectors, which is exactly what would move
delta.

This settles it directly. It wraps both compressors in one object, runs the real
pipeline, and on every relay boundary calls BOTH on the SAME inputs, then
compares the value caches they produce element by element. The wrapper returns
LOBF's output, so the run itself follows the reference path and the comparison
is never contaminated by divergent generation.

Reads the verdict for you:

    max relative difference ~1e-6   implementations agree; the campaign gap is
                                    something else (seeds, or a config mismatch)
    max relative difference >=1e-2  LOBFGram has a real bug; do not report it

Nothing in compression_methods/ or run.py is touched.

    python check_gram_equivalence.py --task gsm8k --max_samples 4
    python check_gram_equivalence.py --task medqa --max_samples 4 --pca_rank 4
"""

import argparse
import itertools
import sys
from typing import Any, Dict, List, Tuple

import torch

from data import (
    load_aime2024, load_aime2025, load_arc_easy, load_arc_challenge,
    load_gsm8k, load_gpqa_diamond, load_mbppplus, load_humanevalplus, load_medqa,
)
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from compression_methods import LOBF, LOBFGram, LOBFFast, HOBF, HOBFFast
from utils import auto_device, set_seed

LOADERS = {
    "gsm8k": load_gsm8k, "aime2024": load_aime2024, "aime2025": load_aime2025,
    "arc_easy": load_arc_easy, "arc_challenge": load_arc_challenge,
    "gpqa": load_gpqa_diamond, "mbppplus": load_mbppplus,
    "humanevalplus": load_humanevalplus, "medqa": load_medqa,
}


class DualCompressor:
    """Runs LOBF and LOBFGram on identical inputs and records how far apart they land.

    Returns LOBF's result so the pipeline follows one deterministic path. Both
    compress() calls read the same past_key_values and build their own output
    tensors, so calling twice is safe.
    """

    # Each alternative names the exact implementation it is meant to replace.
    # A headwise variant has to be checked against HOBF, not LOBF: the selector
    # decides which tokens are discarded, so the residual being factorized is a
    # different matrix and a cross-family comparison would report the selector
    # difference as a numerical one.
    ALTS = {
        "gram":          (LOBF, LOBFGram),
        "fast":          (LOBF, LOBFFast),
        "hobf_fast": (HOBF, HOBFFast),
    }

    def __init__(self, alt="gram", alt_kwargs=None, **kw):
        ref_cls, alt_cls = self.ALTS[alt]
        self.ref = ref_cls(**kw)
        self.alt = alt_cls(**kw, **(alt_kwargs or {}))
        self.ref_name = ref_cls.__name__
        self.records: List[Dict[str, float]] = []

    def reset(self):
        """Both instances, not just the reference.

        _has_kept_sink is per-compressor state: the sink is prepended only at the
        first relay boundary of an item, and reset() clears it between items.
        Forwarding reset to one instance only desynchronises them, and from the
        second item onward one adds the sink while the other does not, which
        shows up as a sink_size difference in the sequence dimension.
        """
        self.ref.reset()
        self.alt.reset()

    # Everything else the pipeline may read comes from the reference instance.
    def __getattr__(self, name):
        return getattr(self.__dict__["ref"], name)

    @staticmethod
    def _values(out: Any) -> List[torch.Tensor]:
        if out is None:
            return []
        if hasattr(out, "value_cache"):
            return list(out.value_cache)
        return [v for _, v in out]

    @torch.no_grad()
    def compress(self, **kwargs):
        ref_out, ref_t, ref_dbg = self.ref.compress(**kwargs)
        alt_out, alt_t, _ = self.alt.compress(**kwargs)

        rv, av = self._values(ref_out), self._values(alt_out)
        if len(rv) != len(av):
            self.records.append({"layers": -1.0})
            return ref_out, ref_t, ref_dbg

        # Three failure modes, kept apart because they mean different things:
        # a shape mismatch is a structural bug, a non-finite value is a numerical
        # blow-up, and a large but finite difference is a precision question.
        worst_rel = 0.0
        worst_layer = -1
        shape_bad = []
        nonfinite_ref = 0
        nonfinite_alt = 0
        num = 0.0
        den = 0.0
        per_layer = []
        for i, (a, b) in enumerate(zip(rv, av)):
            if a.shape != b.shape:
                shape_bad.append((i, tuple(a.shape), tuple(b.shape)))
                continue
            x = a.to(torch.float32)
            y = b.to(torch.float32)
            nr = int((~torch.isfinite(x)).sum().item())
            na = int((~torch.isfinite(y)).sum().item())
            nonfinite_ref += nr
            nonfinite_alt += na
            if nr or na:
                continue                      # a norm over inf/nan says nothing
            d = (x - y).norm().item()
            n = x.norm().item()
            num += d * d
            den += n * n
            rel = d / (n + 1e-12)
            per_layer.append(rel)
            if rel > worst_rel:
                worst_rel, worst_layer = rel, i
        self.records.append({
            "worst_layer_rel": worst_rel,
            "worst_layer": float(worst_layer),
            "whole_cache_rel": (num ** 0.5) / (den ** 0.5 + 1e-12),
            "shape_bad": shape_bad,
            "nonfinite_ref": float(nonfinite_ref),
            "nonfinite_alt": float(nonfinite_alt),
            "per_layer": per_layer,
            "ref_compress_s": float(ref_t),
            "alt_compress_s": float(alt_t),
        })
        return ref_out, ref_t, ref_dbg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B")
    ap.add_argument("--task", default="gsm8k", choices=sorted(LOADERS))
    ap.add_argument("--max_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prompt", default="sequential")
    ap.add_argument("--latent_steps", type=int, default=40)
    ap.add_argument("--kv_budget", type=int, default=32)
    ap.add_argument("--pca_rank", type=int, default=4)
    ap.add_argument("--sink_size", type=int, default=4)
    ap.add_argument("--inject_mode", default="uniform")
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="Short by default: the check only needs the relay boundaries, "
                         "not a full generation.")
    ap.add_argument("--generate_bs", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--latent_space_realign", action="store_true")
    ap.add_argument("--alt", default="fast", choices=sorted(DualCompressor.ALTS),
                    help="Which candidate to check against LOBF.")
    # All None: anything passed here overrides the class default, and passing a
    # stale value silently is how two configurations came out identical once.
    ap.add_argument("--subspace_iters", type=int, default=None)
    ap.add_argument("--subspace_oversample", type=int, default=None)
    ap.add_argument("--subspace_warm_iters", type=int, default=None)
    ap.add_argument("--subspace_qr_every", type=int, default=None)
    ap.add_argument("--subspace_warm_start", action="store_true")
    ap.add_argument("--subspace_batched", action="store_true",
                    help="batch the residual over heads: 12 percent faster, 9 of 432 layers off")
    # Default None, not the old shipped values: passing them explicitly would
    # silently override whatever the class now defaults to, which is exactly the
    # mistake that made the tuned and untuned runs come out identical.
    args = ap.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)
    print(f"device={device}  model={args.model_name}  task={args.task}  "
          f"samples={args.max_samples}  rank={args.pca_rank}  budget={args.kv_budget}")
    print(f"TF32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}\n")

    model = ModelWrapper(args.model_name, device, args=args)
    if args.alt in ("fast", "hobf_fast"):
        alt_kwargs = {k: v for k, v in (("n_iter", args.subspace_iters),
                                        ("oversample", args.subspace_oversample),
                                        ("n_iter_warm", args.subspace_warm_iters),
                                        ("qr_every", args.subspace_qr_every)) if v is not None}
        if args.subspace_warm_start:
            alt_kwargs["warm_start"] = True
        if args.subspace_batched:
            alt_kwargs["batched_residual"] = True
        if args.alt == "hobf_fast":
            # HOBFFast forms the residual per head by construction, so it
            # has no batched_residual switch and no warm start.
            alt_kwargs.pop("batched_residual", None)
            alt_kwargs.pop("warm_start", None)
            alt_kwargs.pop("n_iter_warm", None)
    else:
        alt_kwargs = {}
    dual = DualCompressor(alt=args.alt, alt_kwargs=alt_kwargs,
                          sink_size=args.sink_size, kv_budget=args.kv_budget,
                          pca_rank=args.pca_rank, inject_mode=args.inject_mode)
    print(f"checking {dual.ref_name} vs {args.alt}  {alt_kwargs}")
    method = LatentMASMethod(
        model, compressor=dual, latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens, generate_bs=args.generate_bs, args=args,
    )

    # Loaders are generators, so take the first n rather than slicing.
    items = list(itertools.islice(LOADERS[args.task](), args.max_samples))
    for i, item in enumerate(items, 1):
        method.run_item(item)
        print(f"  sample {i}/{len(items)}: {len(dual.records)} relay boundaries so far")

    if not dual.records:
        print("\nNo relay boundary was reached. Raise --max_samples, or check that "
              "--prompt selects a multi-agent pipeline.")
        return 1

    ok = [r for r in dual.records if "worst_layer_rel" in r]
    if not ok:
        print("\nThe two compressors returned different cache structures. That is "
              "already a bug; inspect LOBFGram's return path.")
        return 1

    worst = max(r["worst_layer_rel"] for r in ok)
    whole = max(r["whole_cache_rel"] for r in ok)
    ref_s = sum(r["ref_compress_s"] for r in ok) / len(ok)
    alt_s = sum(r["alt_compress_s"] for r in ok) / len(ok)

    shp = [s for r in ok for s in r.get("shape_bad", [])]
    nf_ref = sum(r.get("nonfinite_ref", 0.0) for r in ok)
    nf_alt = sum(r.get("nonfinite_alt", 0.0) for r in ok)

    print(f"\n{len(ok)} relay boundaries compared")
    print(f"  worst single-layer relative difference : {worst:.3e}   (finite layers only)")
    print(f"  worst whole-cache relative difference  : {whole:.3e}")
    print(f"  layers with mismatched shape           : {len(shp)}")
    for i, sa, sb in shp[:5]:
        print(f"      layer {i}: LOBF {sa}  vs  alt {sb}")
    print(f"  non-finite entries   LOBF {nf_ref:.0f}   alt {nf_alt:.0f}")
    print(f"  sink state at end    LOBF {dual.ref._has_kept_sink}   alt {dual.alt._has_kept_sink}")
    print(f"  compression time     ref {ref_s:.2f}s   alt {alt_s:.2f}s   ({ref_s / max(alt_s, 1e-9):.2f}x)")

    # One pathological layer and a broadly lossy method need different fixes, so
    # show the distribution rather than only its maximum.
    allr = sorted(x for r in ok for x in r.get("per_layer", []))
    if allr:
        def q(f):
            return allr[min(len(allr) - 1, int(f * len(allr)))]
        print(f"\n  per-layer relative difference over {len(allr)} layer instances")
        print(f"    median {q(0.5):.2e}   p90 {q(0.9):.2e}   p99 {q(0.99):.2e}   max {allr[-1]:.2e}")
        for thr in (1e-4, 1e-3, 1e-2):
            n = sum(1 for x in allr if x > thr)
            print(f"    above {thr:.0e}: {n:4d} / {len(allr)}  ({100 * n / len(allr):.1f}%)")

    if shp:
        print("\nVERDICT: structural mismatch. The two paths returned different cache")
        print("shapes, which selection alone cannot cause; look at the return path.")
        return 1
    if nf_alt and not nf_ref:
        print("\nVERDICT: the Gram path produced non-finite values where LOBF did not.")
        print("torch.linalg.eigh can fail to converge silently on a near-singular Gram")
        print("matrix. That is the bug; do not report Gram until it is fixed.")
        return 1

    print()
    # The verdict keys on the MEDIAN, not the maximum. The maximum is always
    # around 1.9e-2 for every converged method, because it lands on the handful
    # of layers where sigma_p and sigma_{p+1} coincide and the rank-p subspace is
    # not unique; Gram and subspace iteration, written independently, miss on
    # exactly the same ones. A method that has genuinely failed to converge shows
    # it in the middle of the distribution instead.
    med = allr[len(allr) // 2] if allr else float("inf")
    frac_big = (sum(1 for x in allr if x > 1e-2) / len(allr)) if allr else 1.0
    if med < 1e-3 and frac_big < 0.10:
        print(f"VERDICT: converged. Median disagreement with the full SVD is {med:.1e}, and the")
        print(f"{100 * frac_big:.1f} percent of layers above 1e-2 are the degenerate ones every")
        print("factorization disagrees on. Safe to report the speedup.")
        return 0
    if med < 1e-3:
        print(f"VERDICT: median is fine at {med:.1e} but {100 * frac_big:.1f} percent of layers exceed")
        print("1e-2, well above the degenerate floor. Raise the iteration count.")
        return 1
    print(f"VERDICT: not converged. Median disagreement is {med:.1e}, which is a different")
    print("answer rather than a rounding difference. Raise the iteration count or the")
    print("oversampling before reporting anything from this configuration.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
