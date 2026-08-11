"""Numerical equivalence test for the batched OBF injection path.

LOBFBatched replaces LOBF's per-head Python loop with batched torch.linalg
calls. The math per head is unchanged, so the injected delta must agree to
floating-point tolerance. This test reimplements both paths on synthetic
tensors of the shapes the real code produces and compares them directly,
so it runs on CPU in a second and needs no model.

The one thing that legitimately differs is the sign of the singular vectors,
which batched and unbatched LAPACK calls may choose differently. delta is a
projection onto their span, so it is invariant; the test would catch it if
that reasoning were wrong.

    python -m pytest tests/test_lobf_batched_equivalence.py
"""

import torch

EPS = 1e-12
H, K, ND, D = 8, 32, 96, 128   # Qwen3-4B shapes: 8 KV heads, budget 32, head_dim 128


def delta_scalar(V_keep, V_disc, w_disc, scale, pca_rank, center):
    """LOBF's per-head loop, verbatim in structure."""
    out = torch.zeros((V_keep.shape[0], V_keep.shape[-1]), dtype=torch.float32)
    for h in range(V_keep.shape[0]):
        X, Y = V_keep[h], V_disc[h]
        if center:
            mu = X.mean(dim=0, keepdim=True)
            Xc, Yc = X - mu, Y - mu
        else:
            Xc, Yc = X, Y
        Q, _ = torch.linalg.qr(Xc.t(), mode="reduced")
        R = Yc - (Yc @ Q) @ Q.t()
        if torch.linalg.norm(R, ord="fro").item() <= 1e-10:
            continue
        _, S, Vh = torch.linalg.svd(R, full_matrices=False)
        p = min(pca_rank, int(Vh.shape[0]))
        if p <= 0:
            continue
        C = Vh[:p, :]
        wd = w_disc[h]
        wd_sum = wd.sum()
        wd_norm = wd / (wd_sum + EPS) if wd_sum.item() > EPS else torch.full_like(wd, 1.0 / wd.numel())
        r_mean = (wd_norm.unsqueeze(0) @ R).squeeze(0)
        out[h] = ((r_mean @ C.t()) @ C) * scale[h]
    return out


def delta_batched(V_keep, V_disc, w_disc, scale, pca_rank, center):
    """LOBFBatched's head-batched path, verbatim in structure."""
    Hn, Dn = V_keep.shape[0], V_keep.shape[-1]
    out = torch.zeros((Hn, Dn), dtype=torch.float32)
    if center:
        mu = V_keep.mean(dim=1, keepdim=True)
        Xc, Yc = V_keep - mu, V_disc - mu
    else:
        Xc, Yc = V_keep, V_disc
    Q = torch.linalg.qr(Xc.transpose(-2, -1), mode="reduced")[0]
    R = Yc - (Yc @ Q) @ Q.transpose(-2, -1)
    _, S, Vh = torch.linalg.svd(R, full_matrices=False)
    active = torch.linalg.norm(R, ord="fro", dim=(-2, -1)) > 1e-10
    p = min(pca_rank, int(Vh.shape[-2]))
    if p <= 0 or not active.any():
        return out
    C = Vh[:, :p, :]
    wd_sum = w_disc.sum(dim=-1, keepdim=True)
    wd_norm = torch.where(
        wd_sum > EPS,
        w_disc / (wd_sum + EPS),
        torch.full_like(w_disc, 1.0 / float(w_disc.shape[-1])),
    )
    r_mean = torch.bmm(wd_norm.unsqueeze(1), R).squeeze(1)
    coeff = torch.bmm(r_mean.unsqueeze(1), C.transpose(-2, -1)).squeeze(1)
    delta = torch.bmm(coeff.unsqueeze(1), C).squeeze(1) * scale.unsqueeze(-1)
    return torch.where(active.unsqueeze(-1), delta, out)


def _case(seed, pca_rank, center, degenerate=False, zero_weights=False):
    g = torch.Generator().manual_seed(seed)
    V_keep = torch.randn(H, K, D, generator=g, dtype=torch.float32)
    V_disc = torch.randn(H, ND, D, generator=g, dtype=torch.float32)
    if degenerate:
        # One head whose discarded block lies exactly in span(kept): residual is
        # numerically zero, which is the branch the scalar path skips.
        V_disc[0] = V_keep[0][:1].expand(ND, D).clone()
    w_disc = torch.rand(H, ND, generator=g, dtype=torch.float32)
    if zero_weights:
        w_disc[3] = 0.0
    scale = torch.rand(H, generator=g, dtype=torch.float32)
    a = delta_scalar(V_keep, V_disc, w_disc, scale, pca_rank, center)
    b = delta_batched(V_keep, V_disc, w_disc, scale, pca_rank, center)
    return a, b


def test_matches_across_ranks_and_centering():
    for pca_rank in (2, 4, 8, 32):
        for center in (False, True):
            a, b = _case(seed=pca_rank + int(center), pca_rank=pca_rank, center=center)
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-3), (
                f"rank={pca_rank} center={center} max|diff|={(a - b).abs().max():.3e}"
            )


def test_zero_residual_head_is_skipped_identically():
    a, b = _case(seed=11, pca_rank=4, center=False, degenerate=True)
    assert torch.allclose(a[0], torch.zeros(D), atol=1e-5), "scalar path should emit no delta"
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-3), f"max|diff|={(a - b).abs().max():.3e}"


def test_zero_attention_weights_fall_back_to_uniform():
    a, b = _case(seed=12, pca_rank=4, center=False, zero_weights=True)
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-3), f"max|diff|={(a - b).abs().max():.3e}"





# --------------------------------------------------------------------------
# Gram-matrix path (LOBFGram): the top-p right singular vectors of R are the
# top-p eigenvectors of G = R^T R, so this must reproduce the full-SVD delta
# exactly rather than approximately. It also has to reproduce the diagnostics,
# since eigenvalues are S^2 and trace(G) is ||R||_F^2.
# --------------------------------------------------------------------------


def delta_gram(V_keep, V_disc, w_disc, scale, pca_rank, center):
    """LOBFGram's path."""
    Hn, Dn = V_keep.shape[0], V_keep.shape[-1]
    out = torch.zeros((Hn, Dn), dtype=torch.float32)
    if center:
        mu = V_keep.mean(dim=1, keepdim=True)
        Xc, Yc = V_keep - mu, V_disc - mu
    else:
        Xc, Yc = V_keep, V_disc
    Q = torch.linalg.qr(Xc.transpose(-2, -1), mode="reduced")[0]
    R = Yc - (Yc @ Q) @ Q.transpose(-2, -1)
    G = torch.bmm(R.transpose(-2, -1), R)
    lam, vec = torch.linalg.eigh(G)
    lam, vec = lam.flip(-1).clamp_min(0.0), vec.flip(-1)
    active = torch.linalg.norm(R, ord="fro", dim=(-2, -1)) > 1e-10
    p = min(pca_rank, int(lam.shape[-1]))
    if p <= 0 or not active.any():
        return out
    C = vec[:, :, :p].transpose(-2, -1)
    wd_sum = w_disc.sum(dim=-1, keepdim=True)
    wd_norm = torch.where(
        wd_sum > EPS,
        w_disc / (wd_sum + EPS),
        torch.full_like(w_disc, 1.0 / float(w_disc.shape[-1])),
    )
    r_mean = torch.bmm(wd_norm.unsqueeze(1), R).squeeze(1)
    coeff = torch.bmm(r_mean.unsqueeze(1), C.transpose(-2, -1)).squeeze(1)
    delta = torch.bmm(coeff.unsqueeze(1), C).squeeze(1) * scale.unsqueeze(-1)
    return torch.where(active.unsqueeze(-1), delta, out)


def _gram_case(seed, pca_rank, center, nd=488, **kw):
    g = torch.Generator().manual_seed(seed)
    V_keep = torch.randn(H, K, D, generator=g, dtype=torch.float32)
    # Give the residual a decaying spectrum, as real V blocks have; white noise
    # would make every low-rank method look wrong for reasons unrelated to code.
    basis = torch.linalg.qr(torch.randn(H, D, D, generator=g))[0]
    coefs = torch.randn(H, nd, D, generator=g) * (torch.arange(1, D + 1) ** -0.42)
    V_disc = coefs @ basis.transpose(-2, -1)
    if kw.get("degenerate"):
        V_disc[0] = V_keep[0][:1].expand(nd, D).clone()
    w_disc = torch.rand(H, nd, generator=g, dtype=torch.float32)
    if kw.get("zero_weights"):
        w_disc[3] = 0.0
    scale = torch.rand(H, generator=g, dtype=torch.float32)
    a = delta_scalar(V_keep, V_disc, w_disc, scale, pca_rank, center)
    b = delta_gram(V_keep, V_disc, w_disc, scale, pca_rank, center)
    return a, b


def test_gram_matches_full_svd():
    for pca_rank in (2, 4, 8, 32):
        for center in (False, True):
            a, b = _gram_case(seed=100 + pca_rank + int(center), pca_rank=pca_rank, center=center)
            rel = (a - b).norm() / a.norm()
            assert rel < 1e-4, f"rank={pca_rank} center={center} rel={rel:.3e}"


def test_gram_eigenvalues_reproduce_singular_values_squared():
    g = torch.Generator().manual_seed(7)
    R = torch.randn(H, 488, D, generator=g)
    s2 = torch.linalg.svdvals(R) ** 2
    lam = torch.linalg.eigvalsh(torch.bmm(R.transpose(-2, -1), R)).flip(-1).clamp_min(0.0)
    assert torch.allclose(s2, lam, rtol=1e-3, atol=1e-2), (s2[0, :3], lam[0, :3])
    fro2 = torch.linalg.norm(R, ord="fro", dim=(-2, -1)) ** 2
    assert torch.allclose(fro2, lam.sum(-1), rtol=1e-4), "trace(G) must equal ||R||_F^2"


def test_gram_handles_degenerate_and_zero_weight_heads():
    for kw in ({"degenerate": True}, {"zero_weights": True}):
        a, b = _gram_case(seed=200, pca_rank=4, center=False, **kw)
        rel = (a - b).norm() / a.norm()
        assert rel < 1e-4, f"{kw} rel={rel:.3e}"


if __name__ == "__main__":
    test_matches_across_ranks_and_centering()
    test_zero_residual_head_is_skipped_identically()
    test_zero_attention_weights_fall_back_to_uniform()
    test_gram_matches_full_svd()
    test_gram_eigenvalues_reproduce_singular_values_squared()
    test_gram_handles_degenerate_and_zero_weight_heads()
    print("all equivalence tests passed")
