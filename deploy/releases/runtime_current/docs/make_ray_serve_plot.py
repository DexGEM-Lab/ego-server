#!/usr/bin/env python3
"""
Deterministic small-multiples throughput/latency plot from Ray Serve benchmark data.
Reads extracted data inline; all values traced to raw artifacts on dex-a800.

Corrections from raw evidence (2026-07-17):
  - HaWoR: bounded backpressure (max_ongoing_requests=8, max_queued=32); rejection
    fraction plotted; "all admitted/unbounded queue" removed.
  - DROID: grouped scatter/lines per session count; key configurations labeled.
  - UniDepth: stable through 4 img/s; knee between 4 and 6.
  - Hands/WiLoR: N=8/level → p95 descriptive only.
  - Cosmos: achieved>offered explained as finite-window timing; unsupported token
    claims removed; N=5 p95 descriptive.
  - Throughput lower bounds vs latency-SLO limits distinguished throughout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 150,
})

# ── data ──────────────────────────────────────────────────────────────────

# 1. UniDepth (GPU0) — raw: gpu0_unidepth_endpoint_openloop_20260716T200015Z_final/open_loop_sweep/summary.json
ud_offered = [1, 2, 4, 6, 8]
ud_achieved = [1.0085, 2.0137, 3.9877, 4.4075, 4.2313]
ud_p95_ms = [161.995, 161.415, 321.17, 5765.884, 10414.68]
ud_unit = "img/s"
ud_label = "UniDepth\ndepth est."
ud_note = "N=100/level; dynamic batch up to 8 at 8 img/s"

# 2. Hands Detect (GPU1) — raw: gpu1_hands_wilor_20260716T2001Z/open_loop_summary.csv
hd_offered = [1, 4, 8, 16, 32]
hd_achieved = [1, 4, 8, 16, 32]
hd_p95_ms = [283.2, 261.5, 778.5, 1274.7, 1565.9]
hd_unit = "req/s"
hd_label = "Hands.Detect"
hd_note = "N=8/level → p95 descriptive only; no throughput saturation ≤32 req/s"

# 3. WiLoR Reconstruct (GPU1)
wr_offered = [1, 4, 8, 16, 32]
wr_achieved = [1, 4, 8, 16, 32]
wr_p95_ms = [103.3, 102.6, 101.5, 172.7, 268.7]
wr_unit = "req/s"
wr_label = "WiLoR.Reconstruct"
wr_note = "N=8/level → p95 descriptive only; no throughput saturation ≤32 req/s"

# 4. DROID (GPU2) — grouped by session count
# Raw: droid_openloop_final_20260716T204043Z/droid/summary.json
#
# Driver semantics: each level creates N=sessions concurrent sessions. Each session
# receives exactly 8 push_frame calls dispatched at a Poisson rate of waves-per-s
# req/s. Total push_frame count = sessions × 8 per level. The "offered_rate_per_s"
# field is measured as offered_count / elapsed_s (finite-window rate), not the
# target ideal sessions × waves-per-s — actual offered varies due to scheduler
# timing and admission rejection.
#
# x = offered_rate_per_s (from summary.json)
# y = completed_rate_per_s (from summary.json)
# Hollow markers = any rejection (rejected_count > 0)
#
# All 20 push_frame data points (4 session counts × 5 wave rates).

droid_sessions = {
    1: {  # sessions=1, 8 push_frames/level, all completed
        "offered":   [0.5635, 1.1165, 2.1827, 4.1749, 7.7414],
        "achieved":  [0.5635, 1.1165, 2.1827, 4.1749, 7.7414],
        "p95_s":     [0.9289, 0.1993, 0.1961, 0.1941, 0.2273],
        "rejected_frac": [0.0, 0.0, 0.0, 0.0, 0.0],
    },
    2: {  # sessions=2, 16 push_frames/level
        "offered":   [1.1194, 2.1949, 4.2282, 7.8856, 13.3600],
        "achieved":  [1.1194, 2.1949, 4.2282, 7.8856, 6.6800],
        "p95_s":     [0.3785, 0.3756, 0.3719, 0.4356, 0.7966],
        "rejected_frac": [0.0, 0.0, 0.0, 0.0, 0.50],   # 8/16 rejected
    },
    4: {  # sessions=4, 32 push_frames/level
        "offered":   [2.1993, 4.2400, 7.9045, 13.8711, 19.3287],
        "achieved":  [2.1993, 4.2400, 7.6575, 6.9355, 6.0402],
        "p95_s":     [0.7620, 0.7383, 0.9111, 1.2802, 1.3849],
        "rejected_frac": [0.0, 0.0, 0.031, 0.50, 0.688],  # 1/32 at w=2, 16/32 at w=4, 22/32 at w=8
    },
    8: {  # sessions=8, 64 push_frames/level
        "offered":   [4.4311, 8.6353, 13.8422, 19.1877, 19.7630],
        "achieved":  [4.4311, 8.6353, 6.9211, 5.0967, 5.2495],
        "p95_s":     [1.6592, 1.7154, 2.4879, 2.8247, 2.7288],
        "rejected_frac": [0.0, 0.0, 0.50, 0.734, 0.734],  # 32/64 at w=2, 47/64 at w=4, 47/64 at w=8
    },
}

d_unit = "frames/s"
d_label = "DROID\nvisual odometry"
d_note = "8 push_frames/session/level; admission ceiling ~7.5–8.6 f/s; grouped by session count"
d_saturation_ceiling = 8.6

# 5. HaWoR (GPU3) — raw: gpu3_hawor_infiller_20260716T2003Z/open_loop_sweep_100x5_retry.json
# Bounded backpressure: max_ongoing_requests=8 + max_queued_requests=32
hw_offered = [1, 2, 4, 8, 16]
hw_achieved = [1.001, 1.982, 1.027, 0.854, 0.884]
hw_p95_ms = [630.59, 1017.91, 33574.19, 42818.92, 34945.80]
# Outcome counts per level (from raw retry JSON rows):
hw_ok =    [100, 100, 52, 42, 44]   # HTTP 200
hw_error = [0,   0,   48, 58, 56]   # HTTP 503 (queue overflow)
hw_unit = "req/s"
hw_label = "HaWoR\nhand est."
hw_note = "N=100/level; bounded backpressure: 503 errors at ≥4 req/s"

# 6. HaWoR Infiller (GPU3)
hi_offered = [1, 2, 4, 8, 16]
hi_achieved = [1.009, 2.017, 4.027, 8.031, 15.967]
hi_p95_ms = [61.68, 59.72, 59.24, 57.73, 71.93]
hi_unit = "req/s"
hi_label = "HaWoR Infiller"
hi_note = "N=100/level; no saturation ≤16 req/s; ~60 ms baseline"

# 7. Cosmos3 Nano (GPU6) — raw: cosmos3_open_loop_3f980de_20260716T205845Z/cosmos3/open_loop_summary.json
co_offered = [0.25, 0.5, 1.0, 2.0, 4.0]
co_achieved = [0.309, 0.612, 1.199, 1.819, 3.092]
co_p95_ms = [798.98, 427.47, 556.40, 818.62, 866.51]
co_unit = "req/s"
co_label = "Cosmos3 Nano\nLLM inference"
co_note = "N=5/level → p95 descriptive only; achieved>offered: finite-window timing"
# prompt ~2k tokens/req, completion tokens vary (89–251); total 25 reqs, 51,825 prompt tokens

# ── panel definitions ─────────────────────────────────────────────────────

colors = {
    "unidepth":     "#1f77b4",
    "hands":        "#ff7f0e",
    "wilor":        "#2ca02c",
    "droid":        "#d62728",
    "hawor":        "#9467bd",
    "infiller":     "#8c564b",
    "cosmos":       "#e377c2",
}

droid_session_colors = {1: "#d62728", 2: "#ff7f0e", 4: "#2ca02c", 8: "#1f77b4"}
droid_session_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

fig, axes = plt.subplots(3, 3, figsize=(11.5, 10))
fig.suptitle("Ray Serve — Per-API Throughput vs. Latency (Single Replica, 1×A800 each)",
             fontsize=10, fontweight="bold", y=0.98)

color_twin = "#555555"

# ── Panel 0: UniDepth ──
ax = axes[0, 0]
ax.plot(ud_offered, ud_achieved, "o-", color=colors["unidepth"], linewidth=1.3, markersize=4,
        label="Achieved throughput", zorder=5)
max_val = max(max(ud_offered), max(ud_achieved)) * 1.15
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6, label="y = x")
ax.set_xlabel(f"Offered load ({ud_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({ud_unit})", fontsize=7, color=colors["unidepth"])
ax.tick_params(axis="y", labelcolor=colors["unidepth"], labelsize=6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

ax2 = ax.twinx()
p95_s = [p / 1000.0 for p in ud_p95_ms]
ax2.plot(ud_offered, p95_s, "s--", color=color_twin, linewidth=1.0, markersize=3,
         markerfacecolor="white", markeredgecolor=color_twin, label="p95 latency")
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)
ax.set_title(f"{ud_label} (N=100)", fontsize=8, pad=2)

# Annotations: stable through 4, knee between 4 and 6
ax.axvspan(0, 4, alpha=0.06, color="green")
ax.annotate("stable\np95 ≤ 0.32 s", xy=(2, 1.5), fontsize=5.5, color="green", ha="center")
ax.annotate("knee\np95 → 5.77 s", xy=(6, 4.41), xytext=(6.5, 2.5),
            fontsize=6, ha="left", color="grey",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.7))
ax.annotate("queueing\np95 > 10 s", xy=(8, 4.23), xytext=(6, 1.5),
            fontsize=5.5, color="grey", ha="left",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.7))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=5.5, framealpha=0.8)

# ── Panel 1: Hands.Detect ──
ax = axes[0, 1]
ax.plot(hd_offered, hd_achieved, "o-", color=colors["hands"], linewidth=1.3, markersize=4, zorder=5)
max_val = max(max(hd_offered), max(hd_achieved)) * 1.15
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6)
ax.set_xlabel(f"Offered load ({hd_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({hd_unit})", fontsize=7, color=colors["hands"])
ax.tick_params(axis="y", labelcolor=colors["hands"], labelsize=6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax2 = ax.twinx()
p95_s = [p / 1000.0 for p in hd_p95_ms]
ax2.plot(hd_offered, p95_s, "s--", color=color_twin, linewidth=1.0, markersize=3,
         markerfacecolor="white", markeredgecolor=color_twin)
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)
ax.set_title(f"{hd_label} (N=8)", fontsize=8, pad=2)
ax.annotate("throughput: no saturation\np95: descriptive only (N=8)\nlatency → 1.57 s at 32 req/s",
            xy=(30, 30), fontsize=5.5, color="grey", ha="right", va="top")

# ── Panel 2: WiLoR.Reconstruct ──
ax = axes[0, 2]
ax.plot(wr_offered, wr_achieved, "o-", color=colors["wilor"], linewidth=1.3, markersize=4, zorder=5)
max_val = max(max(wr_offered), max(wr_achieved)) * 1.15
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6)
ax.set_xlabel(f"Offered load ({wr_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({wr_unit})", fontsize=7, color=colors["wilor"])
ax.tick_params(axis="y", labelcolor=colors["wilor"], labelsize=6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax2 = ax.twinx()
p95_s = [p / 1000.0 for p in wr_p95_ms]
ax2.plot(wr_offered, p95_s, "s--", color=color_twin, linewidth=1.0, markersize=3,
         markerfacecolor="white", markeredgecolor=color_twin)
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)
ax.set_title(f"{wr_label} (N=8)", fontsize=8, pad=2)
ax.annotate("throughput: no saturation\np95: descriptive only (N=8)\n~100 ms baseline; p95 ≤ 0.27 s",
            xy=(30, 30), fontsize=5.5, color="grey", ha="right", va="top")

# ── Panel 3: DROID (grouped by session count) ──
ax = axes[1, 0]
all_offered = []
all_achieved = []
for s, d in droid_sessions.items():
    off = d["offered"]
    ach = d["achieved"]
    all_offered.extend(off)
    all_achieved.extend(ach)
    c = droid_session_colors[s]
    m = droid_session_markers[s]
    # Connect points within same session group
    ax.plot(off, ach, "-", color=c, linewidth=0.8, alpha=0.5, zorder=3)
    # Mark overload points (rejection > 0) with hollow markers
    for i, (ox, oy) in enumerate(zip(off, ach)):
        rej = d["rejected_frac"][i]
        if rej > 0:
            ax.plot(ox, oy, m, color=c, markersize=6, markerfacecolor="white",
                    markeredgecolor=c, markeredgewidth=1.2, zorder=5)
        else:
            ax.plot(ox, oy, m, color=c, markersize=5, zorder=5)
    ax.plot([], [], f"{m}-", color=c, markersize=5, label=f"s={s}")

max_val = max(max(all_offered), max(all_achieved)) * 1.15
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6)
ax.set_xlabel(f"Offered load ({d_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({d_unit})", fontsize=7, color=colors["droid"])
ax.tick_params(axis="y", labelcolor=colors["droid"], labelsize=6)
ax.axhline(y=d_saturation_ceiling, color="red", linewidth=0.7, linestyle=":", alpha=0.5)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# Right axis: p95 latency (only for non-rejected points, same grouping)
ax2 = ax.twinx()
for s, d in droid_sessions.items():
    # Filter to non-rejected points for latency display
    off = [d["offered"][i] for i in range(len(d["offered"])) if d["rejected_frac"][i] == 0]
    p95 = [d["p95_s"][i] for i in range(len(d["p95_s"])) if d["rejected_frac"][i] == 0]
    if off:
        ax2.plot(off, p95, "s--", color=color_twin, linewidth=0.8, markersize=3,
                 markerfacecolor="white", markeredgecolor=color_twin, alpha=0.6)
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)
ax.set_title(f"{d_label}", fontsize=8, pad=2)

# Key annotation: s=2×4 (7.89 f/s, no reject)
ax.annotate("s=2×4 w/s\n7.89 f/s, p95 .436 s\nno push reject",
            xy=(7.89, 7.89), xytext=(4, 4.5),
            fontsize=5, color="#ff7f0e", ha="left",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.6))
# s=8×1 (8.64 f/s, 1 finalize fail)
ax.annotate("s=8×1 w/s\n8.64 f/s, p95 1.72 s\n1 finalize failure",
            xy=(8.64, 8.64), xytext=(4, 10),
            fontsize=5, color="#1f77b4", ha="left",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.6))
# Saturation / rejection annotation — overload points now at correct offered x
ax.annotate("~50% in-flight reject\ns=2×8, s=4×4, s=8×2\n(offered ~13–14 f/s)",
            xy=(13.36, 6.68), xytext=(15.5, 3),
            fontsize=5, color="grey", ha="left",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.6))
# Admission ceiling
ax.annotate(f"~{d_saturation_ceiling} f/s admission ceiling",
            xy=(d_saturation_ceiling*0.7, d_saturation_ceiling), xytext=(1, d_saturation_ceiling+2),
            fontsize=5.5, color="red", ha="left",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.6))
ax.legend(loc="upper left", fontsize=5.5, framealpha=0.8)

# ── Panel 4: HaWoR (with rejection fraction) ──
ax = axes[1, 1]
hw_color = colors["hawor"]
# Achieved throughput (successful completions only)
ax.plot(hw_offered, hw_achieved, "o-", color=hw_color, linewidth=1.3, markersize=5,
        label="Achieved (successes only)", zorder=5)
max_val = max(max(hw_offered), max(hw_achieved)) * 1.15
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6, label="y = x")
ax.set_xlabel(f"Offered load ({hw_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({hw_unit})", fontsize=7, color=hw_color)
ax.tick_params(axis="y", labelcolor=hw_color, labelsize=6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# Right axis: p95 latency
ax2 = ax.twinx()
p95_s = [p / 1000.0 for p in hw_p95_ms]
ax2.plot(hw_offered, p95_s, "s--", color=color_twin, linewidth=1.0, markersize=3,
         markerfacecolor="white", markeredgecolor=color_twin, label="p95 latency")
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)

# Rejection fraction as bar chart (second twin on same side as throughput but with bars)
# Use stacked labels: annotate each point with ok/error counts
for i, (ox, oy) in enumerate(zip(hw_offered, hw_achieved)):
    ok = hw_ok[i]
    err = hw_error[i]
    if err > 0:
        # Annotate error fraction above the achieved point
        frac = err / (ok + err)
        ax.annotate(f"{err}/{ok+err} err\n({frac:.0%})",
                    xy=(ox, oy), xytext=(0, 12), textcoords="offset points",
                    fontsize=5.5, color="red", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="red", linewidth=0.5))

ax.set_title(f"{hw_label} (N=100)", fontsize=8, pad=2)
# Annotations
ax.annotate("saturation at\n~2 req/s", xy=(2, 1.98), xytext=(3.5, 1.2),
            fontsize=6, ha="left",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
ax.annotate("p95 > 30 s (of successes)\nbounded backpressure\nmax_ongoing=8, max_queued=32",
            xy=(8, 0.85), xytext=(9.5, 0.3),
            fontsize=5.5, color="grey", ha="left",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.7))
# Add legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=5.5, framealpha=0.8)

# ── Panel 5: HaWoR Infiller ──
ax = axes[1, 2]
ax.plot(hi_offered, hi_achieved, "o-", color=colors["infiller"], linewidth=1.3, markersize=4, zorder=5)
max_val = max(max(hi_offered), max(hi_achieved)) * 1.15
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6)
ax.set_xlabel(f"Offered load ({hi_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({hi_unit})", fontsize=7, color=colors["infiller"])
ax.tick_params(axis="y", labelcolor=colors["infiller"], labelsize=6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax2 = ax.twinx()
p95_s = [p / 1000.0 for p in hi_p95_ms]
ax2.plot(hi_offered, p95_s, "s--", color=color_twin, linewidth=1.0, markersize=3,
         markerfacecolor="white", markeredgecolor=color_twin)
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)
ax.set_title(f"{hi_label} (N=100)", fontsize=8, pad=2)
ax.annotate("no saturation\n≤16 req/s", xy=(15, 15), fontsize=6, color="green", ha="right", va="bottom")

# ── Panel 6: Cosmos3 ──
ax = axes[2, 0]
ax.plot(co_offered, co_achieved, "o-", color=colors["cosmos"], linewidth=1.3, markersize=4,
        label="Achieved throughput", zorder=5)
# Annotate achieved>offered: finite-window timing artifact
for ox, oy in zip(co_offered, co_achieved):
    if oy > ox * 1.1:
        ax.annotate(f"{oy:.3f}", (ox, oy), textcoords="offset points",
                    xytext=(3, -8), fontsize=5, color=colors["cosmos"], alpha=0.8)
max_val = max(max(co_offered), max(co_achieved)) * 1.2
lim = max(max_val, 0.1)
ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.7, alpha=0.6, label="y = x")
ax.set_xlabel(f"Offered load ({co_unit})", fontsize=7)
ax.set_ylabel(f"Achieved ({co_unit})", fontsize=7, color=colors["cosmos"])
ax.tick_params(axis="y", labelcolor=colors["cosmos"], labelsize=6)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax2 = ax.twinx()
p95_s = [p / 1000.0 for p in co_p95_ms]
ax2.plot(co_offered, p95_s, "s--", color=color_twin, linewidth=1.0, markersize=3,
         markerfacecolor="white", markeredgecolor=color_twin, label="p95 latency")
ax2.set_ylabel("p95 latency (s)", fontsize=7, color=color_twin)
ax2.tick_params(axis="y", labelcolor=color_twin, labelsize=6)
ax.set_title(f"{co_label} (N=5)", fontsize=8, pad=2)
ax.annotate("achieved > offered:\nfinite-window timing\nartifact at low rates",
            xy=(0.25, 0.31), xytext=(1.5, 0.6),
            fontsize=5.5, color="grey", ha="left",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.7))
ax.annotate("no saturation ≤4 req/s\nN=5 → p95 descriptive only",
            xy=(3.8, 3.1), fontsize=5.5, color="grey", ha="right", va="top")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=5.5, framealpha=0.8)

# ── Panel 7 (axes[2,1]): Throughput vs. Latency-SLO note ──
ax = axes[2, 1]
ax.axis("off")
note_text = (
    "Throughput lower bounds vs. latency-SLO limits\n"
    "─────────────────────────────────────────\n"
    "Each panel shows achieved throughput (solid) as\n"
    "a lower bound on capacity: the system can serve\n"
    "at least this rate. The p95 latency (dashed) is\n"
    "the cost at that rate, not a capacity limit.\n\n"
    "A latency-SLO limit (e.g. \"p95 < 1 s\") is a\n"
    "different question: at what offered load does\n"
    "p95 exceed the SLO? That load may be well below\n"
    "the throughput ceiling.\n\n"
    "Key: UniDepth SLO-capacity ~4 img/s (p95 .32 s);\n"
    "HaWoR SLO-capacity ~2 req/s (p95 1.02 s).\n"
    "DROID admission ceiling ~8 f/s is a concurrency\n"
    "limit, not a processing ceiling."
)
ax.text(0.5, 0.5, note_text, transform=ax.transAxes, fontsize=6.5,
        verticalalignment="center", horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.6, edgecolor="grey", linewidth=0.5))

# ── Panel 8 (axes[2,2]): Data source / legend ──
ax = axes[2, 2]
ax.axis("off")
source_text = (
    "Data sources (dex-a800, 2026-07-16)\n"
    "──────────────────────────────────\n"
    "GPU0: UniDepth (depth estimation)\n"
    "GPU1: Hands.Detect + WiLoR.Reconstruct\n"
    "GPU2: DROID (visual odometry)\n"
    "GPU3: HaWoR + Infiller\n"
    "GPU6: Cosmos3 Nano (vLLM 0.19.1)\n\n"
    "Cosmos3: ~2k prompt tokens/req;\n"
    "completion tokens vary 89–251.\n"
    "5 levels × 5 reqs = 25 total.\n\n"
    "All APIs: single replica per GPU;\n"
    "Ray Serve open-loop Poisson generator."
)
ax.text(0.5, 0.5, source_text, transform=ax.transAxes, fontsize=6.5,
        verticalalignment="center", horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", alpha=0.6, edgecolor="grey", linewidth=0.5))

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = "/home/yiwen/ego_annotation-api-ify/docs/ray_serve_throughput_latency_overview.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")

import os
sz = os.path.getsize(out_path)
print(f"File size: {sz:,} bytes ({sz/1024:.0f} KB)")
from PIL import Image
im = Image.open(out_path)
print(f"Dimensions: {im.size[0]} × {im.size[1]} px")
