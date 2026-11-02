#!/usr/bin/env python3
"""Fit fresh real-vs-fake probes on the features saved by scripts/probe_d_target.py (CPU or GPU, torch only).

For every feature set (source F/T, history noise/cleanGT/own, raw t, block) and pooling:
  global      : token means averaged over the 16 target frames and 2 views (the old D's pooling), 1536-d
  global+std  : global mean ++ global token-std (spatial variability, 'texture'), 3072-d
  frame       : one sample per target frame (views averaged); clip-arm score = mean of its 16 frame logits
  frame+std   : as frame with mean ++ std per frame
probes (all fresh, nothing from the trained head):
  logreg : standardised (train-fold statistics), L2 logistic regression solved by batched Newton; the
           penalty is chosen per outer fold by an inner grouped CV (pooled inner AUC)
  mlp    : standardised, 1 hidden layer (256, GELU, dropout 0.2), AdamW, fixed hyperparameters
Evaluation: CLIP-LEVEL grouped k-fold (both arms and all frames of a clip are always on the same side),
repeated with different fold assignments.  Held-out decision scores are pooled over folds; reported:
AUC (mean over repeats), paired accuracy P(score(real) > score(fake)) within a clip, a clip-bootstrap 95%
CI, and the AUC on train-split / val-split clips separately.
References without fitting: the OLD disc head applied to the F block-12 global mean (old_head), and a
latent-statistics baseline (per-channel mean/std/|dx|/|dy|/|dt| of the noised target latents, logreg).

Outputs in --input-dir: fit_results.jsonl (one line per fit, written as it goes), fit_summary.txt,
fit_results.json.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

POOLINGS = ("global", "global+std", "frame", "frame+std")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--lambdas", default="1e-3,1e-2,1e-1,1,10,100")
    p.add_argument("--poolings", default="global,frame,global+std",
                   help=f"subset of {','.join(POOLINGS)} (frame+std is the slowest; off by default)")
    p.add_argument("--time-budget-min", type=float, default=30.0,
                   help="stop starting new feature sets after this many minutes (partial results are kept)")
    p.add_argument("--mlp-poolings", default="global,frame")
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-steps", type=int, default=300)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-wd", type=float, default=1e-2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--only", default="", help="comma list of substrings; fit only matching feature sets")
    p.add_argument("--min-clips", type=int, default=8)
    p.add_argument("--self-test", action="store_true", help="synthetic checks (solver vs closed forms)")
    p.add_argument("--out-prefix", default="fit")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------
def pair_matrix(s_real, s_fake):
    """M[i, j] = 1[s_real_i > s_fake_j] + 0.5 * 1[==]  (float64)."""
    a = s_real.double().view(-1, 1)
    b = s_fake.double().view(1, -1)
    return (a > b).double() + 0.5 * (a == b).double()


def auc(s_real, s_fake):
    return float(pair_matrix(s_real, s_fake).mean().item())


def bootstrap_auc(mats, n_boot, seed=0):
    """Clip bootstrap of the repeat-averaged AUC.  mats: [R, n, n] pair matrices over the same clips."""
    n = mats.shape[-1]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    w = torch.zeros(n_boot, n, dtype=torch.float64)
    w.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
    vals = torch.einsum("bi,rij,bj->br", w, mats.cpu().double(), w) / float(n * n)
    vals = vals.mean(dim=1)
    lo, hi = torch.quantile(vals, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return lo, hi


def summarize_scores(scores, splits, n_boot):
    """scores: [R, n, 2] held-out clip-arm scores (arm 0 = real, 1 = fake)."""
    R = scores.shape[0]
    mats = torch.stack([pair_matrix(scores[r, :, 0], scores[r, :, 1]) for r in range(R)])
    aucs = mats.mean(dim=(1, 2)).tolist()
    paired = [float((scores[r, :, 0] > scores[r, :, 1]).double().mean().item()) for r in range(R)]
    lo, hi = bootstrap_auc(mats, n_boot) if n_boot > 0 else (float("nan"), float("nan"))
    out = {"auc": sum(aucs) / R, "auc_repeats": [round(a, 4) for a in aucs], "auc_ci95": [lo, hi],
           "paired_acc": sum(paired) / R, "n_clips": int(scores.shape[1]),
           "heldout_scores": [[[round(float(v), 4) for v in arm] for arm in rep] for rep in scores.tolist()]}
    for sp in sorted(set(splits)):
        sel = torch.tensor([s == sp for s in splits])
        if sel.sum() >= 2:
            out[f"auc_{sp}"] = float(mats[:, sel][:, :, sel].mean().item())
            out[f"n_{sp}"] = int(sel.sum().item())
    return out


# ----------------------------------------------------------------------------------
# folds
# ----------------------------------------------------------------------------------
def make_folds(n, k, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(n, generator=g)
    fold = torch.empty(n, dtype=torch.long)
    fold[perm] = torch.arange(n) % k
    return fold


# ----------------------------------------------------------------------------------
# batched L2 logistic regression (Newton, masks select each problem's training samples)
# ----------------------------------------------------------------------------------
def _batched_cg(hvp, g, iters, rtol=1e-6):
    """Solve H_p s_p = g_p for SPD H_p given batched products hvp(V) [P, d] (conjugate gradients)."""
    s = torch.zeros_like(g)
    r = g.clone()
    q = r.clone()
    rs = (r * r).sum(-1)
    g2 = rs.clamp(min=1e-30)
    for _ in range(iters):
        Hq = hvp(q)
        alpha = rs / (q * Hq).sum(-1).clamp(min=1e-30)
        s = s + alpha.unsqueeze(1) * q
        r = r - alpha.unsqueeze(1) * Hq
        rs_new = (r * r).sum(-1)
        if float((rs_new / g2).max()) < rtol ** 2:
            break
        q = r + (rs_new / rs.clamp(min=1e-30)).unsqueeze(1) * q
        rs = rs_new
    return s


def newton_logreg(X, y, masks, lams, iters=30, tol=None, force=None, cg_threshold=768, cg_iters=60):
    """min_w (1/n_p) sum_i m_pi log(1+exp(-yt_i x_i.w)) + lam_p/2 |w|^2 for P problems at once.

    X [N, d] (already standardised, include a constant column for the bias), y [N] in {0,1},
    masks [P, N] in {0,1}, lams [P].  Returns W [P, d].  Uses the d x d Hessian or the N x N Woodbury
    form, whichever needs fewer flops, or matrix-free Newton-CG when both N and d exceed cg_threshold
    (force='primal'/'woodbury'/'cg' overrides).
    """
    X = X.to(torch.float64) if X.device.type == "cpu" else X.float()
    dt = X.dtype
    P, N = masks.shape
    d = X.shape[1]
    yt = (2.0 * y.to(dt) - 1.0)
    m = masks.to(dt)
    n_p = m.sum(dim=1).clamp(min=1.0)
    lam = lams.to(dt).view(P, 1)
    if tol is None:
        tol = 1e-12 if dt == torch.float64 else 1e-7
    woodbury = (N ** 3 / 3.0) < (N * d * d + d ** 3 / 3.0)
    use_cg = min(N, d) > cg_threshold
    if force is not None:
        woodbury, use_cg = force == "woodbury", force == "cg"
    if use_cg:
        woodbury = False
    W = torch.zeros(P, d, dtype=dt, device=X.device)
    K = X @ X.t() if woodbury else None
    eyeN = torch.eye(N, dtype=dt, device=X.device) if woodbury else None
    eyed = None if (woodbury or use_cg) else torch.eye(d, dtype=dt, device=X.device)
    steps = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.02, 0.0], dtype=dt, device=X.device)

    def objective(Wc):          # [S, P, d] -> [S, P]
        z = torch.einsum("nd,spd->spn", X, Wc)
        loss = (torch.nn.functional.softplus(-yt * z) * m).sum(-1) / n_p
        return loss + 0.5 * lam.view(1, P) * Wc.square().sum(-1)

    for _ in range(iters):
        z = W @ X.t()                                   # [P, N]
        pr = torch.sigmoid(z)
        sg = torch.sigmoid(-yt * z)
        g = -(X.t() @ (m * yt * sg).t()).t() / n_p.view(P, 1) + lam * W     # [P, d]
        D = m * pr * (1 - pr) / n_p.view(P, 1)          # [P, N]
        if use_cg:
            step = _batched_cg(lambda V: ((V @ X.t()) * D) @ X + lam * V, g, cg_iters)
        elif not woodbury:
            H = torch.einsum("nd,pn,ne->pde", X, D, X) + lam.view(P, 1, 1) * eyed
            L, info = torch.linalg.cholesky_ex(H)
            step = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
        else:
            sD = D.sqrt()                                # [P, N]
            A = sD.unsqueeze(2) * K.unsqueeze(0) * sD.unsqueeze(1) + lam.view(P, 1, 1) * eyeN
            rhs = sD * (g @ X.t())                       # [P, N]
            L, info = torch.linalg.cholesky_ex(A)
            zsol = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
            step = (g - (sD * zsol) @ X) / lam
        dec = (g * step).sum(-1)
        if float(dec.abs().max()) < tol:
            break
        cand = W.unsqueeze(0) - steps.view(-1, 1, 1) * step.unsqueeze(0)     # [S, P, d]
        f = objective(cand)
        best = f.argmin(dim=0)
        W = cand[best, torch.arange(P, device=X.device)]
    return W


def standardize(X, train_mask):
    mu = X[train_mask].mean(dim=0, keepdim=True)
    sd = X[train_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
    Z = (X - mu) / sd
    return torch.cat([Z, torch.ones_like(Z[:, :1])], dim=1)


def clip_arm_scores(z, sample_clip, sample_arm, n_clips):
    """Average per-sample scores into [n_clips, 2]."""
    out = torch.zeros(n_clips * 2, dtype=z.dtype, device=z.device)
    cnt = torch.zeros(n_clips * 2, dtype=z.dtype, device=z.device)
    idx = sample_clip * 2 + sample_arm
    out.index_add_(0, idx, z)
    cnt.index_add_(0, idx, torch.ones_like(z))
    return (out / cnt.clamp(min=1)).view(n_clips, 2)


def cv_logreg(X, sample_clip, sample_arm, n_clips, args, lams, device):
    """Outer grouped CV with inner grouped CV for lambda.  Returns held-out scores [R, n, 2] and info."""
    X = X.to(device)
    y = (sample_arm == 0).to(device)                    # real = 1
    sc = sample_clip.to(device)
    sa = sample_arm.to(device)
    L = len(lams)
    k_out = min(args.folds, n_clips)
    scores = torch.zeros(args.repeats, n_clips, 2, dtype=torch.float64)
    chosen = []
    for r in range(args.repeats):
        fold = make_folds(n_clips, k_out, r).to(device)
        for k in range(k_out):
            test_clip = fold == k
            train_s = ~test_clip[sc]
            Z = standardize(X, train_s)
            train_clips = torch.nonzero(~test_clip).flatten()
            k_in = min(args.inner_folds, len(train_clips))
            inner = torch.full((n_clips,), -1, dtype=torch.long, device=device)
            g = torch.Generator().manual_seed(7 + 31 * r + k)
            perm = train_clips[torch.randperm(len(train_clips), generator=g).to(device)]
            inner[perm] = torch.arange(len(perm), device=device) % k_in
            masks, lam_list = [], []
            for j in range(k_in):
                mj = (inner[sc] >= 0) & (inner[sc] != j)
                for lam in lams:
                    masks.append(mj)
                    lam_list.append(lam)
            for lam in lams:
                masks.append(train_s)
                lam_list.append(lam)
            M = torch.stack(masks).to(torch.float64 if device == "cpu" else torch.float32)
            W = newton_logreg(Z, y, M, torch.tensor(lam_list, device=device))
            zall = W @ Z.t().to(W.dtype)                # [P, N]
            # inner AUC per lambda (pooled over inner folds, clip-arm level)
            inner_auc = []
            for li in range(L):
                s_in = torch.zeros(n_clips, 2, dtype=torch.float64, device=device)
                for j in range(k_in):
                    pidx = j * L + li
                    ca = clip_arm_scores(zall[pidx].double(), sc, sa, n_clips)
                    sel = inner == j
                    s_in[sel] = ca[sel]
                tr = inner >= 0
                inner_auc.append(auc(s_in[tr, 0], s_in[tr, 1]))
            best = max(range(L), key=lambda i: (round(inner_auc[i], 6), i))   # ties -> stronger penalty
            chosen.append(lams[best])
            ca = clip_arm_scores(zall[k_in * L + best].double(), sc, sa, n_clips)
            scores[r][test_clip.cpu()] = ca[test_clip].cpu()
    return scores, {"lambda_chosen": chosen}


def cv_mlp(X, sample_clip, sample_arm, n_clips, args, device):
    X = X.to(device).float()
    y = (sample_arm == 0).float().to(device)
    sc = sample_clip.to(device)
    sa = sample_arm.to(device)
    k_out = min(args.folds, n_clips)
    scores = torch.zeros(args.repeats, n_clips, 2, dtype=torch.float64)
    h = args.mlp_hidden
    for r in range(args.repeats):
        fold = make_folds(n_clips, k_out, r).to(device)
        Zs, masks = [], []
        for k in range(k_out):
            train_s = ~(fold == k)[sc]
            Zs.append(standardize(X, train_s)[:, :-1])
            masks.append(train_s.float())
        Z = torch.stack(Zs)                              # [F, N, d]
        M = torch.stack(masks)                           # [F, N]
        Fn, N, d = Z.shape
        g = torch.Generator(device="cpu").manual_seed(11 + r)
        W1 = (torch.randn(Fn, d, h, generator=g) / math.sqrt(d)).to(device).requires_grad_(True)
        b1 = torch.zeros(Fn, 1, h, device=device, requires_grad=True)
        W2 = (torch.randn(Fn, h, 1, generator=g) / math.sqrt(h)).to(device).requires_grad_(True)
        b2 = torch.zeros(Fn, 1, 1, device=device, requires_grad=True)
        opt = torch.optim.AdamW([W1, b1, W2, b2], lr=args.mlp_lr, weight_decay=args.mlp_wd)
        drop = torch.nn.Dropout(0.2)
        n_tr = M.sum(dim=1)
        for _ in range(args.mlp_steps):
            hid = drop(torch.nn.functional.gelu(torch.bmm(Z, W1) + b1))
            z = (torch.bmm(hid, W2) + b2).squeeze(-1)    # [F, N]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                z, y.expand_as(z), reduction="none") * M).sum(dim=1) / n_tr
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            hid = torch.nn.functional.gelu(torch.bmm(Z, W1) + b1)
            z = (torch.bmm(hid, W2) + b2).squeeze(-1)
            for k in range(k_out):
                test_clip = fold == k
                ca = clip_arm_scores(z[k].double(), sc, sa, n_clips)
                scores[r][test_clip.cpu()] = ca[test_clip].cpu()
    return scores, {}


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def load_clips(input_dir):
    files = sorted((Path(input_dir) / "clips").glob("*.pt"))
    clips = []
    for f in files:
        try:
            clips.append(torch.load(f, map_location="cpu", weights_only=False))
        except Exception as exc:  # noqa: BLE001
            print(f"[fit] skip unreadable {f.name}: {exc!r}", flush=True)
    return clips, [f.stem for f in files]


def feature_sets(clips):
    keys = set(clips[0]["mean"])
    for c in clips[1:]:
        keys &= set(c["mean"])
    sets = []
    for key in sorted(keys):
        src, hist, arm, t = key.split("|")
        if arm != "fake":
            continue
        real = f"{src}|{'cleanGT' if hist == 'own' else hist}|real|{t}"
        if real in keys:
            sets.append({"src": src, "hist": hist, "t": int(t[1:]), "real": real, "fake": key})
    order = {"noise": 0, "cleanGT": 1, "own": 2}
    sets.sort(key=lambda s: (order.get(s["hist"], 9), s["src"], s["t"]))
    return sets


def build_samples(real, fake, pooling):
    """real/fake: [n, 16, V, C'] tensors (C' = C or 2C when std is concatenated).  Returns X, clip, arm."""
    n = real.shape[0]
    both = torch.stack([real, fake], dim=1).float()      # [n, 2, 16, V, C']
    if pooling.startswith("global"):
        X = both.mean(dim=(2, 3)).reshape(n * 2, -1)
        clip = torch.arange(n).repeat_interleave(2)
        arm = torch.tensor([0, 1]).repeat(n)
    else:
        fr = both.mean(dim=3)                            # [n, 2, 16, C']
        T = fr.shape[2]
        X = fr.reshape(n * 2 * T, -1)
        clip = torch.arange(n).repeat_interleave(2 * T)
        arm = torch.tensor([0, 1]).repeat_interleave(T).repeat(n)
    return X, clip, arm


def fmt_table(rows, cols, cell):
    w = max(len(r) for r in rows) + 2
    head = " " * w + "".join(f"{c:>12s}" for c in cols)
    lines = [head]
    for r in rows:
        lines.append(f"{r:<{w}s}" + "".join(f"{cell(r, c):>12s}" for c in cols))
    return lines


# ----------------------------------------------------------------------------------
def self_test(device):
    """Solver correctness: KKT residual of the Newton solution, primal vs Woodbury agreement."""
    torch.manual_seed(0)
    rep = {}
    for N, d in ((60, 200), (300, 40)):
        X = torch.randn(N, d)
        w_true = torch.randn(d) / math.sqrt(d)
        y = (torch.sigmoid(3 * X @ w_true) > torch.rand(N)).float()
        Z = standardize(X.to(device), torch.ones(N, dtype=torch.bool, device=device))
        masks = torch.ones(3, N, device=device)
        masks[1, : N // 3] = 0
        lams = torch.tensor([1e-2, 1e-1, 1.0], device=device)
        W = newton_logreg(Z, y.to(device), masks, lams)
        Zd = Z.to(W.dtype)
        yt = 2 * y.to(device).to(W.dtype) - 1
        res = []
        for p in range(3):
            m = masks[p].to(W.dtype)
            g = -(Zd.t() @ (m * yt * torch.sigmoid(-yt * (Zd @ W[p])))) / m.sum() + lams[p] * W[p]
            res.append(float(g.abs().max()))
        rep[f"N{N}_d{d}_max_kkt_residual"] = max(res)
    # Woodbury vs primal on the same problems (force both paths), with masks and several lambdas
    for N, d in ((80, 79), (50, 120)):
        X = torch.randn(N, d)
        y = (torch.rand(N) > 0.5).float()
        Z = standardize(X, torch.ones(N, dtype=torch.bool))
        masks = torch.ones(2, N)
        masks[1, :7] = 0
        lams = torch.tensor([0.05, 0.5])
        W1 = newton_logreg(Z, y, masks, lams, force="primal")
        W2 = newton_logreg(Z, y, masks, lams, force="woodbury")
        rep[f"N{N}_d{d}_primal_vs_woodbury_max_abs_diff"] = float((W1 - W2).abs().max())
    X = torch.randn(300, 120)
    y = (torch.sigmoid(X[:, :5].sum(1)) > torch.rand(300)).float()
    Z = standardize(X, torch.ones(300, dtype=torch.bool))
    masks = torch.ones(3, 300)
    masks[2, :40] = 0
    lams = torch.tensor([1e-2, 1e-1, 1.0])
    W1 = newton_logreg(Z, y, masks, lams, force="primal")
    W2 = newton_logreg(Z, y, masks, lams, force="cg", cg_iters=200)
    rep["N300_d120_primal_vs_cg_rel_diff"] = float((W1 - W2).abs().max() / W1.abs().max())
    rep["ok"] = all(v < 1e-4 for k, v in rep.items() if k != "ok")
    return rep


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    st = self_test(device)
    print(f"[fit] solver self-test on {device}: {json.dumps(st)}", flush=True)
    if args.self_test:
        return 0 if st["ok"] else 2
    if not st["ok"]:
        print("[fit] solver self-test FAILED; refusing to fit", flush=True)
        return 2
    t0 = time.time()
    in_dir = Path(args.input_dir)
    clips, names = load_clips(in_dir)
    args._clip_names = names
    if len(clips) < args.min_clips:
        print(f"[fit] only {len(clips)} clips; need {args.min_clips}", flush=True)
        return 1
    blocks = clips[0]["blocks"]
    n = len(clips)
    splits = [c["meta"]["split"] for c in clips]
    lams = [float(x) for x in args.lambdas.split(",")]
    poolings = [p for p in args.poolings.split(",") if p]
    mlp_pool = [p for p in args.mlp_poolings.split(",") if p]
    only = [s for s in args.only.split(",") if s]
    has_std = bool(clips[0]["std"])
    sets = feature_sets(clips)
    t_info = clips[0]["t_info"]
    print(f"[fit] {n} clips ({splits.count('train')} train / {splits.count('val')} val), "
          f"{len(sets)} feature sets x {len(blocks)} blocks, std={has_std}, device={device}", flush=True)
    jl_path = in_dir / f"{args.out_prefix}_results.jsonl"
    jl = open(jl_path, "w")
    results = []

    def emit(rec):
        results.append(rec)
        jl.write(json.dumps(rec) + "\n")
        jl.flush()

    # old head (no fitting)
    for s in sets:
        if s["src"] != "F":
            continue
        try:
            sr = torch.tensor([c["old_head_logit"][s["real"]] for c in clips])
            sf = torch.tensor([c["old_head_logit"][s["fake"]] for c in clips])
        except KeyError:
            continue
        scores = torch.stack([sr, sf], dim=1).unsqueeze(0).double()
        rec = {"kind": "old_head", "src": "F", "hist": s["hist"], "t": s["t"], "block": 12,
               "pooling": "global", "model": "old_head",
               "mean_logit_real": float(sr.mean()), "mean_logit_fake": float(sf.mean()),
               "mean_paired_diff": float((sr - sf).mean())}
        rec.update(summarize_scores(scores, splits, args.bootstrap))
        emit(rec)
    # latent-statistics baseline
    lat_keys = sorted({k.split("|", 1)[1] for k in clips[0]["latstats"]}, key=lambda k: (k != "clean", k))
    for lk in lat_keys:
        real = torch.stack([c["latstats"][f"real|{lk}"] for c in clips]).unsqueeze(2)   # [n, 16, 1, 80]
        fake = torch.stack([c["latstats"][f"fake|{lk}"] for c in clips]).unsqueeze(2)
        t = 0 if lk == "clean" else int(lk[1:])
        for pooling in ("global", "frame"):
            X, sclip, sarm = build_samples(real, fake, pooling)
            scores, info = cv_logreg(X, sclip, sarm, n, args, lams, device)
            rec = {"kind": "latstats", "src": "latent", "hist": "-", "t": t, "block": -1,
                   "pooling": pooling, "model": "logreg", **info}
            rec.update(summarize_scores(scores, splits, args.bootstrap))
            emit(rec)
    print(f"[fit] references done in {time.time() - t0:.0f} s", flush=True)

    # DiT feature probes, in priority passes so a time-limited run still has the key tables
    passes = [("global", "logreg"), ("global", "mlp"), ("frame", "logreg"), ("global+std", "logreg"),
              ("frame", "mlp"), ("frame+std", "logreg")]
    passes = [(pl, m) for pl, m in passes if pl in poolings and (m == "logreg" or pl in mlp_pool)
              and (has_std or not pl.endswith("+std"))]
    hist_order = {"noise": 0, "cleanGT": 1, "own": 2}
    out_of_time = False
    for pi, (pooling, model) in enumerate(passes):
        for si, s in enumerate(sorted(sets, key=lambda q: hist_order.get(q["hist"], 9))):
            tag = f"{s['src']}|{s['hist']}|t{s['t']}"
            if only and not any(o in tag for o in only):
                continue
            if (time.time() - t0) / 60.0 > args.time_budget_min:
                out_of_time = True
                break
            stat = "std" if pooling.endswith("+std") else None
            mr = torch.stack([c["mean"][s["real"]] for c in clips])    # [n, B, 16, V, C] fp16
            mf = torch.stack([c["mean"][s["fake"]] for c in clips])
            if stat:
                sr = torch.stack([c["std"][s["real"]] for c in clips])
                sf = torch.stack([c["std"][s["fake"]] for c in clips])
            for bi, b in enumerate(blocks):
                if stat:
                    real = torch.cat([mr[:, bi], sr[:, bi]], dim=-1)
                    fake = torch.cat([mf[:, bi], sf[:, bi]], dim=-1)
                else:
                    real, fake = mr[:, bi], mf[:, bi]
                X, sclip, sarm = build_samples(real, fake, pooling)
                tm = time.time()
                if model == "logreg":
                    scores, info = cv_logreg(X, sclip, sarm, n, args, lams, device)
                else:
                    scores, info = cv_mlp(X, sclip, sarm, n, args, device)
                ti = t_info.get(s["t"]) or t_info.get(str(s["t"])) or {}
                rec = {"kind": "probe", "src": s["src"], "hist": s["hist"], "t": s["t"], "block": b,
                       "pooling": pooling, "model": model, "sigma": ti.get("sigma"),
                       "seconds": round(time.time() - tm, 2), **info}
                rec.update(summarize_scores(scores, splits, args.bootstrap))
                emit(rec)
            del mr, mf
        print(f"[fit] pass {pi + 1}/{len(passes)} ({pooling}, {model}) done, {time.time() - t0:.0f} s elapsed",
              flush=True)
        write_summary(in_dir, args, results, blocks, t_info, n, splits, st, time.time() - t0, quiet=True)
        if out_of_time:
            print(f"[fit] time budget {args.time_budget_min} min reached during pass {pi + 1}; stopping", flush=True)
            break
    jl.close()
    write_summary(in_dir, args, results, blocks, t_info, n, splits, st, time.time() - t0)
    return 0


def write_summary(in_dir, args, results, blocks, t_info, n, splits, st, secs, quiet=False):
    def sig(t):
        if t == 0:
            return "clean"
        info = t_info.get(t) or t_info.get(str(t)) or {}
        return f"{info.get('sigma', float('nan')):.3f}"

    L = [f"History-blind D probe: {n} clips ({splits.count('train')} train / {splits.count('val')} val); "
         f"{args.folds}-fold clip-grouped CV x {args.repeats} repeats; inner {args.inner_folds}-fold for lambda; "
         f"fit time {secs / 60:.1f} min", f"solver self-test: {json.dumps(st)}",
         "cells: held-out AUC real-vs-fake (0.5 = chance); CI = clip bootstrap 95%",
         "t_raw -> sigma: " + ", ".join(f"{t}->{sig(t)}" for t in sorted({r['t'] for r in results if r['t']}))]
    probes = [r for r in results if r["kind"] == "probe"]
    for hist in ("noise", "cleanGT", "own"):
        ts = sorted({r["t"] for r in probes if r["hist"] == hist})
        if not ts:
            continue
        for pooling in POOLINGS:
            for model in ("logreg", "mlp"):
                sub = {(r["src"], r["block"], r["t"]): r for r in probes
                       if r["hist"] == hist and r["pooling"] == pooling and r["model"] == model}
                if not sub:
                    continue
                L.append("")
                L.append(f"== hist={hist}  pooling={pooling}  model={model}  (columns: t_raw / sigma)")
                rows = [f"{src} blk{b:02d}" for src in ("F", "T") for b in blocks
                        if any((src, b, t) in sub for t in ts)]
                cols = [f"{t}/{sig(t)}" for t in ts]

                def cell(row, col, sub=sub):
                    src, b = row.split()[0], int(row.split()[1][3:])
                    t = int(col.split("/")[0])
                    r = sub.get((src, b, t))
                    return "-" if r is None else f"{r['auc']:.3f}"
                L.extend(fmt_table(rows, cols, cell))
    L.append("")
    L.append("== best block per (hist, src, pooling, model, t): AUC [CI95] paired_acc  (max over blocks is "
             "optimistic by selection)")
    keyset = sorted({(r["hist"], r["src"], r["pooling"], r["model"], r["t"]) for r in probes})
    for key in keyset:
        cand = [r for r in probes if (r["hist"], r["src"], r["pooling"], r["model"], r["t"]) == key]
        b = max(cand, key=lambda r: r["auc"])
        L.append(f"   {key[0]:7s} {key[1]} {key[2]:10s} {key[3]:6s} t{key[4]:<4d} sigma {sig(key[4])}: "
                 f"blk{b['block']:02d} AUC {b['auc']:.3f} [{b['auc_ci95'][0]:.3f},{b['auc_ci95'][1]:.3f}] "
                 f"paired {b['paired_acc']:.2f}  train {b.get('auc_train', float('nan')):.3f} "
                 f"val {b.get('auc_val', float('nan')):.3f}")
    L.append("")
    L.append("== references")
    for r in results:
        if r["kind"] == "old_head":
            L.append(f"   old head (F blk12, trained with clean history) hist={r['hist']:7s} t{r['t']:<4d} "
                     f"sigma {sig(r['t'])}: AUC {r['auc']:.3f} [{r['auc_ci95'][0]:.3f},{r['auc_ci95'][1]:.3f}] "
                     f"paired {r['paired_acc']:.2f}  logit real {r['mean_logit_real']:+.2f} "
                     f"fake {r['mean_logit_fake']:+.2f}")
    for r in results:
        if r["kind"] == "latstats":
            L.append(f"   latent-stats logreg {r['pooling']:6s} t{r['t']:<4d} sigma {sig(r['t'])}: "
                     f"AUC {r['auc']:.3f} [{r['auc_ci95'][0]:.3f},{r['auc_ci95'][1]:.3f}] "
                     f"paired {r['paired_acc']:.2f}")
    (in_dir / f"{args.out_prefix}_summary.txt").write_text("\n".join(L) + "\n")
    tmp = in_dir / f".{args.out_prefix}_results.json.tmp"
    tmp.write_text(json.dumps({"args": vars(args), "n_clips": n, "splits": splits,
                               "clip_names": getattr(args, "_clip_names", None), "results": results}))
    os.replace(tmp, in_dir / f"{args.out_prefix}_results.json")
    if not quiet:
        print("\n".join(L), flush=True)


if __name__ == "__main__":
    sys.exit(main())
