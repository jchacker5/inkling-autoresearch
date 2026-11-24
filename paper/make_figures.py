"""Build every figure and every number in the paper from the raw results. No number in paper.tex is typed by hand.

    uv run --group paper paper/make_figures.py

Inputs:  results/ledger.tsv, results/eval-*.json, ~/.cache/inkling-autoresearch/*_progress.tsv
Outputs: paper/figures/*.pdf, paper/generated/numbers.tex (\\V{key} macros), paper/generated/*.tex tables,
         results/progress/*.tsv (copies of the training curves, so they are versioned with the paper)
"""
import csv
import json
import shutil
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "inkling-autoresearch"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "generated"
RES = ROOT / "results"
KEPT = ("keep", "candidate")   # swarm verdicts that count as kept (round-2 rule: "candidate")
for d in (FIG, GEN, RES / "progress"):
    d.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (light surface); identity never relies on colour alone: markers + labels.
C = {"looped": "#2a78d6", "control-4L": "#eb6834", "control-10L": "#1baf7a", "extra": "#eda100", "extra2": "#e87ba4"}
MARK = {"looped": "o", "control-4L": "s", "control-10L": "^", "extra": "D", "extra2": "v"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
LABEL = {"looped": "Looped (7.3M, 4 loops)", "control-4L": "Plain 4-layer (same params)",
         "control-10L": "Plain 10-layer (same depth, 2.4x params)"}
LANES = ["recipe", "depth-attn", "halting", "kv-reuse", "history"]
LANE_C = dict(zip(LANES, [C["looped"], C["control-4L"], C["control-10L"], C["extra"], C["extra2"]]))
LANE_M = dict(zip(LANES, ["o", "s", "^", "D", "v"]))

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

VALS = {}


def V(key, value, fmt="{:.3f}"):
    VALS[key] = fmt.format(value) if not isinstance(value, str) else value


def pct(x):
    return f"{100 * x:.1f}\\%"


# ---------------------------------------------------------------- data
def read_tsv(p):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def progress(name):
    src = CACHE / f"{name}_progress.tsv"
    if src.exists():
        shutil.copyfile(src, RES / "progress" / f"{name}.tsv")
    rows = read_tsv(RES / "progress" / f"{name}.tsv")
    out = []
    for r in rows:
        t = float(r.get("train_min") or r.get("wall_min"))
        d = dict(t=t, steps=int(r["steps"]), bpb=float(r["val_bpb"]))
        d.update({h: float(r[f"hop{h}"]) for h in (0, 1, 2, 4, 6, 8)})
        out.append(d)
    return out


def evals():
    out = {}
    for p in sorted(RES.glob("eval-*.json")):
        out[p.stem[5:]] = json.loads(p.read_text())
    return out


ledger = read_tsv(RES / "ledger.tsv")
valid = [r for r in ledger if r["valid"] == "True"]
base_rows = [r for r in valid if r["lane"] == "baseline"]
E = evals()
curves = {k: progress(n) for k, n in (("looped", "base"), ("control-4L", "control-4L"), ("control-10L", "control-10L"))}


def f(r, k):
    return float(r[k])


# ---------------------------------------------------------------- numbers: base + calibration
b = curves["looped"]
if b:
    last = b[-1]
    V("base-hours", last["t"] / 60, "{:.1f}")
    V("base-steps", last["steps"], "{:,}")
    V("base-bpb-quick", last["bpb"])
    for h in (0, 1, 2, 4, 6, 8):
        V(f"base-hop{h}", pct(last[h]))
    first = b[0]
    V("base-first-min", first["t"], "{:.0f}")
    V("base-first-hop1", pct(first[1]))

# ---------------------------------------------------------------- numbers: baseline noise
if base_rows:
    for k in ("val_bpb", "reason_id", "reason_ood", "ood_at_2x", "bpb_at_2x"):
        v = [f(r, k) for r in base_rows]
        V(f"bl-{k}-mean", st.mean(v), "{:.4f}" if "bpb" in k else "{:.3f}")
        V(f"bl-{k}-std", st.stdev(v) if len(v) > 1 else 0.0, "{:.4f}")
    V("bl-n", len(base_rows), "{:d}")
    V("bl-steps", st.mean(int(r["steps"]) for r in base_rows), "{:.0f}")
    V("bl-kv", f(base_rows[0], "kv_kb"), "{:.1f}")
    V("bl-latency", st.mean(f(r, "latency_ms") for r in base_rows), "{:.1f}")
    V("bl-peak", st.mean(f(r, "peak_gb") for r in base_rows), "{:.1f}")
    V("bl-params", int(base_rows[0]["params"]) / 1e6, "{:.2f}M")

# ---------------------------------------------------------------- numbers + table: swarm
agent_rows = [r for r in ledger if r["lane"] in LANES]
V("swarm-n", len(agent_rows), "{:d}")
V("swarm-valid", sum(r["valid"] == "True" for r in agent_rows), "{:d}")
V("swarm-keeps", sum(r["verdict"] in KEPT for r in agent_rows), "{:d}")
V("swarm-lanes", len({r["lane"] for r in agent_rows}), "{:d}")
if agent_rows:
    V("swarm-start", agent_rows[0]["time"][9:11] + ":" + agent_rows[0]["time"][11:13], "{}")
    V("swarm-end", agent_rows[-1]["time"][9:11] + ":" + agent_rows[-1]["time"][11:13], "{}")

bl = {k: st.mean(f(r, k) for r in base_rows) for k in ("val_bpb", "reason_id", "reason_ood")} if base_rows else None
bl_sd = {k: (st.stdev(f(r, k) for r in base_rows) if len(base_rows) > 1 else float("nan"))
         for k in ("val_bpb", "reason_id", "reason_ood")} if base_rows else None


def esc(s):
    return s.replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("$", "\\$").replace("^", "")[:60]


lines = []
confirmed = []
for lane in LANES:
    rs = [r for r in agent_rows if r["lane"] == lane]
    if not rs:
        continue
    kept = [r for r in rs if r["verdict"] in KEPT and r["valid"] == "True"]
    conf = [r for r in valid if r["lane"] == f"{lane}-confirm"]
    if kept and bl:
        best = kept[-1]
        seeds = [best] + conf
        m = {k: st.mean(f(r, k) for r in seeds) for k in ("val_bpb", "reason_id", "reason_ood")}
        sd = {k: (st.stdev(f(r, k) for r in seeds) if len(seeds) > 1 else float("nan")) for k in m}
        # promotion rule from program.md: mean beats baseline by more than 2x the seed std, on some metric,
        # without the other metrics getting worse by more than their noise
        gain_bpb = bl["val_bpb"] - m["val_bpb"]
        gain_id = m["reason_id"] - bl["reason_id"]
        gain_ood = m["reason_ood"] - bl["reason_ood"]
        tol = {k: 2 * max(bl_sd[k], sd[k] if sd[k] == sd[k] else 0) for k in m}
        ok = len(seeds) >= 3 and (gain_bpb > tol["val_bpb"] or gain_id > tol["reason_id"] or gain_ood > tol["reason_ood"]) \
            and gain_bpb > -tol["val_bpb"] and gain_id > -tol["reason_id"] and gain_ood > -tol["reason_ood"]
        if ok:
            confirmed.append(lane)
        lines.append(f"{lane} & {len(rs)} & {len(kept)} & {esc(best['desc'])} & "
                     f"{-gain_bpb:+.4f} & {gain_id:+.3f} & {gain_ood:+.3f} & {len(seeds)} & {'yes' if ok else 'no'} \\\\")
        V(f"{lane}-best", esc(best["desc"]), "{}")
        V(f"{lane}-dbpb", -gain_bpb, "{:+.4f}")
        V(f"{lane}-did", gain_id, "{:+.3f}")
        V(f"{lane}-dood", gain_ood, "{:+.3f}")
        V(f"{lane}-confirmed", "yes" if ok else "no", "{}")
    else:
        lines.append(f"{lane} & {len(rs)} & 0 & -- & & & & & no \\\\")
    V(f"{lane}-n", len(rs), "{:d}")
    V(f"{lane}-keeps", len(kept), "{:d}")
V("confirmed-n", len(confirmed), "{:d}")
V("confirmed-lanes", ", ".join(confirmed) if confirmed else "none", "{}")
(GEN / "lanes.tex").write_text("\n".join(lines) + "\n")

# full ledger appendix table
app = []
for i, r in enumerate(agent_rows, 1):
    app.append(f"{i} & {r['lane']} & {esc(r['desc'])[:46]} & {r['val_bpb'][:6]} & {r['reason_id'][:5]} & "
               f"{r['reason_ood'][:5]} & {r['verdict'] if r['valid'] == 'True' else 'invalid'} \\\\")
(GEN / "ledger.tex").write_text("\n".join(app) + "\n")

# ---------------------------------------------------------------- numbers + table: controls
ctrl_lines = []
for key, name in (("looped", "base-2h"), ("control-4L", "control-4L-2h"), ("control-10L", "control-10L-2h"), ("looped6", "base")):
    e = E.get(name)
    if not e:
        continue
    acc = {int(k): v for k, v in e["reason_by_hop"].items()}
    V(f"{name}-bpb", e["val_bpb"], "{:.4f}")
    V(f"{name}-rid", pct(e["reason_id"]))
    V(f"{name}-rood", pct(e["reason_ood"]))
    V(f"{name}-params", e["params"] / 1e6, "{:.2f}M")
    V(f"{name}-mb", e["weights_mb_bf16"], "{:.1f}")
    V(f"{name}-kv", e["kv_kb_per_token"] or 0, "{:.1f}")
    V(f"{name}-lat", e["latency_ms_b1"], "{:.1f}")
    V(f"{name}-steps", e.get("steps") or 0, "{:,}")
    label = {"looped": "Looped, 4 loops (2 h)", "control-4L": "Plain 4-layer (2 h)",
             "control-10L": "Plain 10-layer (2 h)", "looped6": "Looped, 4 loops (6 h)"}[key]
    ctrl_lines.append(f"{label} & {e['params']/1e6:.2f} & {e['weights_mb_bf16']:.1f} & {e['kv_kb_per_token'] or 0:.1f} & "
                      f"{e['latency_ms_b1']:.1f} & {e['val_bpb']:.3f} & " + " & ".join(f"{100*acc.get(h, 0):.1f}" for h in (2, 4, 6, 8, 10)) + " \\\\")
(GEN / "controls.tex").write_text("\n".join(ctrl_lines) + "\n")
if "base-2h" in E and "control-4L-2h" in E:
    V("gain-ood-vs-4L", 100 * (E["base-2h"]["reason_ood"] - E["control-4L-2h"]["reason_ood"]), "{:+.1f}")
    V("gain-id-vs-4L", 100 * (E["base-2h"]["reason_id"] - E["control-4L-2h"]["reason_id"]), "{:+.1f}")
    V("dbpb-vs-4L", E["base-2h"]["val_bpb"] - E["control-4L-2h"]["val_bpb"], "{:+.3f}")
if "base-2h" in E and "control-10L-2h" in E:
    V("gap-ood-vs-10L", 100 * (E["base-2h"]["reason_ood"] - E["control-10L-2h"]["reason_ood"]), "{:+.1f}")
    V("gap-id-vs-10L", 100 * (E["base-2h"]["reason_id"] - E["control-10L-2h"]["reason_id"]), "{:+.1f}")
    V("dbpb-vs-10L", E["base-2h"]["val_bpb"] - E["control-10L-2h"]["val_bpb"], "{:+.3f}")
    V("mem-ratio-10L", E["control-10L-2h"]["params"] / E["base-2h"]["params"], "{:.1f}")

# loop sweep numbers
for name in ("base", "base-2h"):
    sw = E.get(name, {}).get("sweep")
    if sw:
        for k, s in sw.items():
            a = s["acc_by_hop"]
            V(f"{name}-sweep{k}-bpb", s["val_bpb"], "{:.3f}")
            V(f"{name}-sweep{k}-deep", pct(st.mean(a[h] for h in ("8", "10", "12"))))
            V(f"{name}-sweep{k}-shallow", pct(st.mean(a[h] for h in ("1", "2", "3"))))

# ---------------------------------------------------------------- figures
def finish(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


# Fig 1: training curves at matched wall clock
if b:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for key, cv in curves.items():
        if not cv:
            continue
        t = [p["t"] / 60 for p in cv]
        kw = dict(color=C[key], marker=MARK[key], markevery=max(1, len(cv) // 8), label=LABEL[key])
        axes[0].plot(t, [p["bpb"] for p in cv], **kw)
        axes[1].plot(t, [st.mean(p[h] for h in (2, 4, 6)) for p in cv], **kw)
        axes[2].plot(t, [p[8] for p in cv], **kw)
    axes[0].set_title("Held-out text, bits/byte (lower better)", fontsize=8)
    axes[1].set_title("Tracking, 2-6 hops (seen)", fontsize=8)
    axes[2].set_title("Tracking, 8 hops (unseen)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("training time (h)")
    for ax in axes[1:]:
        ax.axhline(0.1, color=MUTED, lw=0.8, ls=":")
        ax.text(ax.get_xlim()[1], 0.1, " chance", color=MUTED, va="center", ha="left", fontsize=7)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    finish(fig, "curves")

# Fig 2: the swarm trajectory (Karpathy-style progress plot), one panel per metric
if agent_rows and bl:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, k, title in ((axes[0], "val_bpb", "val bpb (lower better)"), (axes[1], "reason_ood", "reasoning, 8-10 hops")):
        ax.axhspan(bl[k] - 2 * bl_sd[k], bl[k] + 2 * bl_sd[k], color=GRID, lw=0)
        ax.axhline(bl[k], color=MUTED, lw=0.8)
        ax.text(0.5, bl[k], "baseline ±2σ", color=MUTED, fontsize=7, va="bottom")
        for lane in LANES:
            pts = [(i, f(r, k), r["verdict"]) for i, r in enumerate(agent_rows, 1) if r["lane"] == lane and r["valid"] == "True"]
            if not pts:
                continue
            ax.scatter([p[0] for p in pts if p[2] not in KEPT], [p[1] for p in pts if p[2] not in KEPT], s=14,
                       facecolors="none", edgecolors=LANE_C[lane], marker=LANE_M[lane], linewidths=0.9)
            ax.scatter([p[0] for p in pts if p[2] in KEPT], [p[1] for p in pts if p[2] in KEPT], s=26,
                       color=LANE_C[lane], marker=LANE_M[lane], label=lane, edgecolors="white", linewidths=0.6)
        ax.set_title(title)
        ax.set_xlabel("experiment # (agent swarm, chronological)")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles = [plt.Line2D([], [], color=LANE_C[l], marker=LANE_M[l], ls="", label=l) for l in LANES if any(r["lane"] == l for r in agent_rows)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.text(0.99, -0.08, "filled = kept, hollow = discarded", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    finish(fig, "swarm")

# Fig 3: accuracy by hop at matched training time
hops = (2, 4, 6, 8, 10)
have = [(k, n) for k, n in (("looped", "base-2h"), ("control-4L", "control-4L-2h"), ("control-10L", "control-10L-2h")) if n in E]
if have:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for key, name in have:
        acc = E[name]["reason_by_hop"]
        ys = [acc[str(h)] if str(h) in acc else acc[h] for h in hops]
        ax.plot(hops, ys, color=C[key], marker=MARK[key], label=LABEL[key])
    ax.axvspan(7, 11, color=GRID, lw=0, zorder=0)
    ax.text(9, ax.get_ylim()[1] * 0.97, "unseen depths", ha="center", va="top", fontsize=7, color=MUTED)
    ax.axhline(0.1, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("pointer hops")
    ax.set_xticks(hops)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Accuracy by depth, 2 h of training each")
    ax.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    fig.tight_layout()
    finish(fig, "hops")

# Fig 4: test-time loop sweep (same weights, more or fewer loops)
sw = E.get("base", {}).get("sweep")
if sw:
    ks = sorted(int(k) for k in sw)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    groups = (("1-3 hops", ("1", "2", "3"), C["looped"], "o"), ("4-6 hops", ("4", "6"), C["control-4L"], "s"),
              ("8-12 hops (unseen)", ("8", "10", "12"), C["control-10L"], "^"))
    for lab, hs, col, mk in groups:
        axes[0].plot(ks, [st.mean(sw[str(k)]["acc_by_hop"][h] for h in hs) for k in ks], color=col, marker=mk, label=lab)
    axes[1].plot(ks, [sw[str(k)]["val_bpb"] for k in ks], color=C["looped"], marker="o")
    for ax in axes:
        ax.axvline(4, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("loops at inference (trained with 4)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_title("Variable tracking accuracy")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Held-out text (bits/byte)")
    fig.tight_layout()
    finish(fig, "sweep")

# Fig 5: intelligence per megabyte: reasoning vs weight memory
pts = [(k, n) for k, n in (("looped", "base-2h"), ("control-4L", "control-4L-2h"), ("control-10L", "control-10L-2h")) if n in E]
if pts:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for key, name in pts:
        e = E[name]
        ax.scatter(e["weights_mb_bf16"], e["reason_ood"], color=C[key], marker=MARK[key], s=40, zorder=3)
        ax.annotate(LABEL[key].split(" (")[0], (e["weights_mb_bf16"], e["reason_ood"]), textcoords="offset points",
                    xytext=(6, -3), fontsize=7, color=INK)
    ax.set_xlabel("weights (MB, bf16)")
    ax.set_ylabel("accuracy, 8-10 hops")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Deep reasoning per megabyte")
    xs = [E[n]["weights_mb_bf16"] for _, n in pts]
    ax.set_xlim(0, max(xs) * 1.45)
    fig.tight_layout()
    finish(fig, "pareto")

# ---------------------------------------------------------------- write macros
out = ["% generated by paper/make_figures.py; do not edit", "\\makeatletter"]
for k, v in sorted(VALS.items()):
    out.append(f"\\@namedef{{v@{k}}}{{{v}}}")
out += ["\\makeatother", "\\newcommand{\\V}[1]{\\ifcsname v@#1\\endcsname\\csname v@#1\\endcsname\\else\\textbf{??#1}\\fi}"]
(GEN / "numbers.tex").write_text("\n".join(out) + "\n")
print(f"{len(VALS)} numbers, figures: {sorted(p.name for p in FIG.glob('*.pdf'))}")
