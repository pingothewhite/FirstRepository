"""Summary figure: static-leverage frontier vs the dynamic rules, in history,
in the Monte Carlo, and in the full withdrawal plan (v6 engine)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def tests(name):
    d = pd.read_csv(os.path.join(RES, name))
    d.columns = ["test", "name"] + list(d.columns[2:])
    return d


T = tests("final_recommended_tests.csv")
PICK = [  # (label, source name, colour, marker, size)
    ("Your ladder: 1.3 → 1.5 at −10%", "U1 ladder 1.3->1.5@-10%", "#eb6834", "s", 60),
    ("Your ladder: −5% / −10% steps", "U2 ladder 1.3->1.4@-5%->1.5@-10%", "#eb6834", "D", 50),
    ("Deep dip (−20%/−25%)", "DD deep dip only 1.3 +0.4@-20% +0.7@-25% (52w)", "#1baf7a", "o", 60),
    ("Vol target 32%", "VT32 only (1.0-1.6x)", "#2a78d6", "^", 60),
    ("Vol target + deep dip (recommended)", "R1 VT32 + 0.4@-20% + 0.3@-25% (52w)", "#2a78d6", "*", 260),
    ("… + RSI(2) boost", "R4 R1 + RSI2 boost 0.4", "#2a78d6", "P", 70),
]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlesize": 11})

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))


def frontier(ax, test, x="Ulcer", y="CAGR", lab_L=(1.0, 1.3, 1.4, 1.6, 2.0), offsets=None, ladder_xy=None):
    d = T[T.test == test].set_index("name")
    st = d[d.index.str.startswith("static")].copy()
    st["L"] = [float(n.split()[1]) for n in st.index]
    st = st[(st.L >= 0.9) & (st.L <= 2.1)].sort_values("L")
    ax.plot(st[x] * 100, st[y] * 100, color=INK2, lw=2, marker="o", ms=4, zorder=2,
            label="Flat leverage (1.0x … 2.0x)")
    for L in lab_L:
        r = st[np.isclose(st.L, L)].iloc[0]
        ax.annotate(f"{L:.1f}x", (r[x] * 100, r[y] * 100), textcoords="offset points", xytext=(6, -12),
                    fontsize=8.5, color=INK2)
    ladder_pts = []
    for (lab, src, c, m, s) in PICK:
        r = d.loc[src]
        ax.scatter(r[x] * 100, r[y] * 100, color=c, marker=m, s=s, zorder=4, edgecolor=SURF, linewidth=1.2)
        if lab.startswith("Your ladder"):
            ladder_pts.append((r[x] * 100, r[y] * 100))
            continue
        dx, dy, ha = (offsets or {}).get(lab, (8, 4, "left"))
        ax.annotate(lab, (r[x] * 100, r[y] * 100), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, color=INK, ha=ha, va="center")
    if ladder_pts and ladder_xy is not None:
        for (px, py) in ladder_pts:
            ax.annotate("", (px, py), xytext=ladder_xy, textcoords="data",
                        arrowprops=dict(arrowstyle="-", color="#eb6834", lw=0.9, shrinkA=0, shrinkB=4))
        ax.text(ladder_xy[0], ladder_xy[1], "Your ladders (1.3→1.5 at −10%;\n−5%/−10% steps): same as flat ~1.45x",
                fontsize=8.5, color=INK, ha="left", va="top")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


frontier(axes[0], "hist full", ladder_xy=(19.6, 16.6),
         offsets={"Deep dip (−20%/−25%)": (0, 12, "center"), "Vol target 32%": (-9, 0, "right"),
                  "Vol target + deep dip (recommended)": (4, 14, "left"), "… + RSI(2) boost": (9, 0, "left")})
axes[0].set_title("History, Dec 1999 – Sep 2026\n(one real path, incl. 2000–02, 2008, 2022)")
axes[0].set_xlabel("Typical drawdown depth (Ulcer index) → more pain")
axes[0].set_ylabel("Annual return (CAGR)")

frontier(axes[1], "MC v5 (post-2005, 21d)", ladder_xy=(24.6, 13.25),
         offsets={"Deep dip (−20%/−25%)": (-9, 0, "right"), "Vol target 32%": (9, 0, "left"),
                  "Vol target + deep dip (recommended)": (0, 15, "center"), "… + RSI(2) boost": (9, 0, "left")})
axes[1].set_title("Monte Carlo, your v5 generator\n(2,000 × 10-year paths, median return)")
axes[1].set_xlabel("Average drawdown depth (Ulcer index) → more pain")
axes[1].set_ylabel("Median annual return")

# ---- plan outcomes (v6 engine, margin + outside debt ON)
v6 = pd.read_csv(os.path.join(RES, "v6_modes.csv"), index_col=0)
ax = axes[2]
st = v6.loc[["static 1.3", "static 1.4", "static 1.5"]]
ax.plot(st.ruin, st.median_end / 1e6, color=INK2, lw=2, marker="o", ms=5, label="Flat leverage")
for n_, r in st.iterrows():
    ax.annotate(n_.replace("static ", "") + "x", (r.ruin, r.median_end / 1e6), textcoords="offset points",
                xytext=(0, 9), fontsize=8.5, color=INK2, ha="center")
plan_pick = [
    ("Deep dip only", "deep dip only: 1.3 +0.4 at -20%, +0.3 more at -25% (52w)", "#1baf7a", "o", 60, (9, 0, "left")),
    ("Vol target 32%", "vol target 32% (1.0-1.6x)", "#2a78d6", "^", 60, (9, 0, "left")),
    ("Vol target + deep dip (recommended)", "RECOMMENDED vol target 32% + deep dip (52w)", "#2a78d6", "*", 260, (11, 0, "left")),
    ("Safer: vol target 28% + deep dip", "safer: vol target 28% + deep dip (52w)", "#2a78d6", "v", 60, (9, 0, "left")),
    ("… + RSI(2) boost", "RECOMMENDED + RSI2 boost 0.4", "#2a78d6", "P", 70, (9, 0, "left")),
]
for lab, src, c, m, s_, off in plan_pick:
    r = v6.loc[src]
    ax.scatter(r.ruin, r.median_end / 1e6, color=c, marker=m, s=s_, zorder=4, edgecolor=SURF, linewidth=1.2)
    ax.annotate(lab, (r.ruin, r.median_end / 1e6), textcoords="offset points", xytext=off[:2],
                fontsize=8.5, color=INK, ha=off[2], va="center")
lad_xy = (11.0, 0.972)
for src, m in [("your ladder 1.3 -> 1.5 at -10%", "s"), ("your ladder 1.3 -> 1.4 at -5% -> 1.5 at -10%", "D")]:
    r = v6.loc[src]
    ax.scatter(r.ruin, r.median_end / 1e6, color="#eb6834", marker=m, s=55, zorder=4, edgecolor=SURF, linewidth=1.2)
    ax.annotate("", (r.ruin, r.median_end / 1e6), xytext=lad_xy, textcoords="data",
                arrowprops=dict(arrowstyle="-", color="#eb6834", lw=0.9, shrinkA=0, shrinkB=4))
ax.text(lad_xy[0], lad_xy[1], "Your ladders: same as flat 1.4x", fontsize=8.5, color=INK, ha="left", va="top")
ax.set_xlim(8.0, 15.0)
ax.set_ylim(0.955, 1.30)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}M"))
ax.set_title("Your plan: \\$760k, \\$6.5k/mo, margin waterfall ON\n(v6 engine, same 2,000 paths, year 10)")
ax.set_xlabel("Chance of running out of money (ruin) → worse")
ax.set_ylabel("Median ending equity")

for a in axes[:2]:
    a.legend(loc="lower right", frameon=False, fontsize=8.5)
axes[2].legend(loc="upper right", frameon=False, fontsize=8.5)
fig.suptitle("Up-and-left of the gray line = better than flat leverage at the same risk", fontsize=13,
             fontweight="bold", color=INK, y=1.02)
plt.tight_layout()
out = os.path.join(HERE, "leverage_rules_summary.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
