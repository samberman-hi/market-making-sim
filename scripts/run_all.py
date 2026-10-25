"""Run every experiment, save figures to figures/ and tables to results/.

    python scripts/run_all.py            (about 15 seconds on a laptop)
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmsim import (  # noqa: E402
    AdaptiveToxicityQuoter,
    AvellanedaStoikov,
    AvellanedaStoikovLimits,
    MarketParams,
    SymmetricQuoter,
    average_as_spread,
    break_even_half_spread,
    expected_pnl_rate,
    simulate,
)

FIG, RES = ROOT / "figures", ROOT / "results"
N_PATHS = 2000
SEED = 2008  # one seed for everything: common random numbers across strategies

# Paper parameters (Avellaneda & Stoikov 2008, section 4)
PAPER = MarketParams(s0=100.0, T=1.0, sigma=2.0, dt=0.005, A=140.0, k=1.5)
GAMMAS = [0.01, 0.1, 0.5]
PAPER_TABLE = {  # gamma: (inventory mean, std, q std), (symmetric mean, std, q std)
    0.01: ((66.78, 8.76, 4.70), (67.36, 13.40, 8.65)),
    0.1: ((62.94, 5.89, 2.80), (67.21, 13.43, 8.66)),
    0.5: ((33.92, 4.72, 1.88), (66.20, 14.53, 8.67)),
}

# ---- plotting style ------------------------------------------------------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  saved figures/{name}")


def as_strategy(gamma, market=PAPER, **kw):
    return AvellanedaStoikov(gamma, market.sigma, market.k, market.T, **kw)


# ---- (i) + (ii): replicate the Avellaneda-Stoikov table, P&L histograms ----------
def experiment_paper_table():
    print("\n(i) Avellaneda-Stoikov (2008) replication, %d paths" % N_PATHS)
    rows, results = [], {}
    for g in GAMMAS:
        inv = simulate(as_strategy(g, name="Inventory"), PAPER, N_PATHS, SEED)
        # The paper's symmetric benchmark (and its "spread" column) uses (2/g) ln(1 + g/k).
        sym_spread = (2 / g) * np.log1p(g / PAPER.k)
        sym = simulate(SymmetricQuoter(sym_spread / 2), PAPER, N_PATHS, SEED)
        results[g] = (inv, sym)
        for label, res, paper in [("Inventory", inv, PAPER_TABLE[g][0]),
                                  ("Symmetric", sym, PAPER_TABLE[g][1])]:
            s = res.summary()
            rows.append({"gamma": g, "strategy": label, "avg spread": s["avg spread"],
                         "mean P&L": s["mean P&L"], "std P&L": s["std P&L"],
                         "mean final q": s["mean final q"], "std final q": s["std final q"],
                         "paper mean P&L": paper[0], "paper std P&L": paper[1],
                         "paper std q": paper[2]})
    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))
    table.round(3).to_csv(RES / "as_replication.csv", index=False)

    # Same-average-spread benchmark (fairer when gamma is large): report for reference
    g = 0.5
    sym_fair = simulate(SymmetricQuoter(average_as_spread(g, 2, 1.5, 1, PAPER.dt) / 2),
                        PAPER, N_PATHS, SEED).summary()
    print(f"  note: symmetric with the inventory strategy's true average spread at gamma=0.5 "
          f"({sym_fair['avg spread']:.2f}): mean {sym_fair['mean P&L']:.1f}, std {sym_fair['std P&L']:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    bins = np.linspace(0, 120, 61)
    for ax, g in zip(axes, GAMMAS):
        inv, sym = results[g]
        ax.hist(sym.final_pnl, bins=bins, color=C[1], alpha=0.55, label="Symmetric")
        ax.hist(inv.final_pnl, bins=bins, color=C[0], alpha=0.75, label="Inventory (A-S)")
        ax.set_title(f"γ = {g}")
        ax.set_xlabel("Terminal P&L")
        for res, col, y in [(inv, C[0], 0.92), (sym, C[1], 0.80)]:
            ax.text(0.02, y, f"μ={res.final_pnl.mean():.1f}  σ={res.final_pnl.std():.1f}",
                    transform=ax.transAxes, color=INK, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=col, lw=1.5))
    axes[0].set_ylabel("Number of paths")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Inventory-aware quoting keeps most of the P&L with a fraction of the risk",
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold", y=1.03)
    save(fig, "pnl_distributions.png")
    return table


# ---- (iii): one sample path ---------------------------------------------------------
def experiment_sample_path():
    print("\n(iii) Sample path")
    res = simulate(as_strategy(0.1), PAPER, n_paths=1, seed=7)
    t, n = res.time, len(res.time) - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(t[:-1], res.ask[0], color=C[1], lw=1.2, label="Ask quote")
    ax1.plot(t[:-1], res.bid[0], color=C[0], lw=1.2, label="Bid quote")
    ax1.plot(t, res.mid[0], color=INK, lw=1.6, label="Mid price", zorder=4)
    buys, sells = res.bid_fills[0] > 0, res.ask_fills[0] > 0
    ax1.scatter(t[:n][buys], res.bid[0][buys], marker="^", s=30, color=C[0],
                edgecolor=SURFACE, linewidth=0.8, zorder=3, label="We buy (bid hit)")
    ax1.scatter(t[:n][sells], res.ask[0][sells], marker="v", s=30, color=C[1],
                edgecolor=SURFACE, linewidth=0.8, zorder=3, label="We sell (ask lifted)")
    ax1.set_ylabel("Price")
    ax1.set_title("Avellaneda-Stoikov quotes (γ = 0.1): quotes skew to push inventory back to zero")
    ax1.legend(ncol=3, loc="best", fontsize=8.5)
    ax2.step(t, res.inventory[0], where="post", color=C[2])
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_ylabel("Inventory q")
    ax2.set_xlabel("Time t")
    fig.tight_layout()
    save(fig, "sample_path.png")


# ---- (iv): risk-aversion frontier ----------------------------------------------------
def experiment_frontier():
    print("\n(iv) Sharpe vs risk aversion")
    gammas = np.geomspace(0.003, 3.0, 13)
    rows = []
    for g in gammas:
        inv = simulate(as_strategy(g), PAPER, N_PATHS, SEED).summary()
        sym = simulate(SymmetricQuoter(inv["avg spread"] / 2), PAPER, N_PATHS, SEED).summary()
        rows.append({"gamma": g, "A-S mean": inv["mean P&L"], "A-S std": inv["std P&L"],
                     "A-S Sharpe": inv["Sharpe"], "Sym mean": sym["mean P&L"],
                     "Sym std": sym["std P&L"], "Sym Sharpe": sym["Sharpe"]})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(RES / "frontier.csv", index=False)
    best = df.loc[df["A-S Sharpe"].idxmax()]
    print(f"  best Sharpe {best['A-S Sharpe']:.2f} at gamma={best['gamma']:.3f} "
          f"(symmetric with same spread: {best['Sym Sharpe']:.2f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(df["A-S std"], df["A-S mean"], "-o", color=C[0], ms=5, label="Avellaneda-Stoikov")
    ax1.plot(df["Sym std"], df["Sym mean"], "-o", color=C[1], ms=5,
             label="Symmetric, same average spread")
    for _, r in df.iloc[::3].iterrows():
        ax1.annotate(f"γ={r['gamma']:.2g}", (r["A-S std"], r["A-S mean"]), xytext=(6, -10),
                     textcoords="offset points", fontsize=8, color=INK2)
    ax1.set_xlabel("Std of terminal P&L (risk)")
    ax1.set_ylabel("Mean terminal P&L")
    ax1.set_title("Risk-return frontier")
    ax1.legend(loc="lower right")
    ax2.semilogx(df["gamma"], df["A-S Sharpe"], "-o", color=C[0], ms=5, label="Avellaneda-Stoikov")
    ax2.semilogx(df["gamma"], df["Sym Sharpe"], "-o", color=C[1], ms=5,
                 label="Symmetric, same average spread")
    ax2.set_xlabel("Risk aversion γ (log scale)")
    ax2.set_ylabel("Sharpe = mean / std of terminal P&L")
    ax2.set_title("Sharpe ratio vs risk aversion")
    ax2.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "risk_aversion_frontier.png")
    return df


# ---- (v): informed traders ---------------------------------------------------------------
def strategies_for(market):
    g, q_max = 0.1, 5
    naive_half = average_as_spread(g, market.sigma, market.k, market.T, market.dt) / 2
    return [
        AvellanedaStoikov(g, market.sigma, market.k, market.T, name="Avellaneda-Stoikov"),
        SymmetricQuoter(naive_half, name="Naive symmetric"),
        AvellanedaStoikovLimits(g, market.sigma, market.k, market.T, q_max, name="A-S + limits"),
        AdaptiveToxicityQuoter(g, market.sigma, market.k, market.T, q_max, name="Adaptive (toxicity)"),
    ]


def experiment_informed():
    print("\n(v) Adverse selection: informed-trader fraction pi (J = %.1f)" % PAPER.jump_size)
    pis = np.round(np.arange(0.0, 0.401, 0.05), 2)
    rows, decomp = [], []
    for pi in pis:
        m = MarketParams(pi=pi)
        for strat in strategies_for(m):
            res = simulate(strat, m, N_PATHS, SEED)
            s = res.summary()
            rows.append({"pi": pi, "strategy": s["strategy"], "mean P&L": s["mean P&L"],
                         "se": s["std P&L"] / np.sqrt(N_PATHS), "std P&L": s["std P&L"],
                         "Sharpe": s["Sharpe"], "avg spread": s["avg spread"],
                         "informed share": s["informed share"]})
            if np.isclose(pi, 0.2):
                decomp.append({"strategy": s["strategy"], **res.decomposition(),
                               "total": s["mean P&L"]})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(RES / "informed_sweep.csv", index=False)
    dec = pd.DataFrame(decomp)
    dec.round(3).to_csv(RES / "pnl_decomposition_pi0.2.csv", index=False)
    print(df.pivot(index="pi", columns="strategy", values="mean P&L").round(1).to_string())
    print("\n  P&L decomposition at pi = 0.2:")
    print(dec.round(1).to_string(index=False))

    names = [s.name for s in strategies_for(PAPER)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name, col in zip(names, C):
        d = df[df.strategy == name]
        ax.fill_between(d.pi, d["mean P&L"] - 1.96 * d.se, d["mean P&L"] + 1.96 * d.se,
                        color=col, alpha=0.18, lw=0)
        ls = "--" if name == "A-S + limits" else "-"
        ax.plot(d.pi, d["mean P&L"], ls, marker="o", ms=5, color=col, label=name)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("Fraction of informed order flow π")
    ax.set_ylabel("Mean terminal P&L (95% CI band)")
    ax.set_title("Fixed-spread quoters get picked off; the toxicity-aware quoter stays profitable")
    ax.legend(loc="lower left")
    save(fig, "informed_traders.png")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    comps = ["spread capture", "adverse selection", "inventory carry"]
    x, w = np.arange(len(dec)), 0.24
    for j, (comp, col) in enumerate(zip(comps, C)):
        ax.bar(x + (j - 1) * w, dec[comp], width=w - 0.02, color=col, label=comp.capitalize())
    ax.scatter(x, dec["total"], marker="D", s=55, color=INK, zorder=3, label="Total P&L")
    for xi, tot in zip(x, dec["total"]):
        ax.annotate(f"{tot:.1f}", (xi, tot), xytext=(10, -3), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.axhline(0, color=INK, lw=1)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x, dec["strategy"])
    ax.set_ylabel("Mean P&L contribution")
    ax.set_title("Where the money goes at π = 0.2: spread earned vs. lost to informed traders")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "pnl_decomposition.png")
    return df, dec


# ---- (vi): break-even spread -------------------------------------------------------------
def experiment_break_even():
    print("\n(vi) Break-even half-spread vs pi")
    pis = np.round(np.arange(0.05, 0.401, 0.05), 2)
    deltas = np.round(np.arange(0.0, 1.601, 0.05), 3)
    n = 1000
    rows, curves = [], {}
    for pi in pis:
        m = MarketParams(pi=pi)
        means = np.array([simulate(SymmetricQuoter(d), m, n, SEED).final_pnl.mean() for d in deltas])
        curves[pi] = means
        first = np.argmax(means > 0)  # first grid point with positive mean P&L
        d0, d1, p0, p1 = deltas[first - 1], deltas[first], means[first - 1], means[first]
        sim_be = d0 - p0 * (d1 - d0) / (p1 - p0)  # linear interpolation of the zero crossing
        rows.append({"pi": pi, "theory": break_even_half_spread(m), "simulated": sim_be,
                     "Glosten-Milgrom pi*J": pi * m.jump_size})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RES / "break_even.csv", index=False)
    print(df.round(3).to_string(index=False))

    naive_half = average_as_spread(0.1, 2, 1.5, 1, PAPER.dt) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    grid = np.linspace(0.001, 0.6, 200)
    theory = [break_even_half_spread(MarketParams(pi=p)) for p in grid]
    ax1.plot(grid, theory, color=C[0], label=r"Theory: $(1-\pi)\,e^{-k\delta}\delta = \pi\,(J-\delta)$")
    ax1.plot(df.pi, df.simulated, "o", color=C[0], ms=7, markeredgecolor=SURFACE,
             label="Simulated (zero crossing of mean P&L)")
    ax1.plot(grid, grid * PAPER.jump_size, "--", color=C[1], lw=1.5,
             label=r"Glosten-Milgrom, inelastic demand: $\delta = \pi J$")
    ax1.axhline(naive_half, color=INK2, lw=1, ls=":")
    ax1.text(0.005, naive_half + 0.03, f"naive quoter's half-spread {naive_half:.2f}",
             fontsize=8.5, color=INK2)
    ax1.axhline(PAPER.jump_size, color=INK2, lw=1, ls=":")
    ax1.text(0.005, PAPER.jump_size - 0.08, "J (informed traders' edge)", fontsize=8.5, color=INK2)
    ax1.set_xlabel("Fraction of informed order flow π")
    ax1.set_ylabel("Break-even half-spread δ*")
    ax1.set_title("Spread needed to cover losses to informed traders")
    ax1.legend(loc="lower right")

    fine = np.linspace(0, 1.6, 300)
    for pi, col in zip([0.1, 0.2, 0.3], C[1:]):
        m = MarketParams(pi=pi)
        ax2.plot(fine, expected_pnl_rate(fine, m) * m.T, color=col, lw=1.6, label=f"π = {pi} theory")
        ax2.plot(deltas, curves[pi], "o", color=col, ms=4.5, markeredgecolor=SURFACE)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_xlabel("Half-spread δ of a symmetric quoter")
    ax2.set_ylabel("Mean terminal P&L")
    ax2.set_title("Expected P&L vs quoted spread (dots = simulation)")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "break_even_spread.png")
    return df


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    start = time.time()
    table = experiment_paper_table()
    experiment_sample_path()
    experiment_frontier()
    informed, _ = experiment_informed()
    experiment_break_even()

    print("\n" + "=" * 72 + "\nSUMMARY")
    inv = table[(table.gamma == 0.1) & (table.strategy == "Inventory")].iloc[0]
    sym = table[(table.gamma == 0.1) & (table.strategy == "Symmetric")].iloc[0]
    print(f"A-S gamma=0.1: mean {inv['mean P&L']:.1f} std {inv['std P&L']:.1f} | "
          f"symmetric: mean {sym['mean P&L']:.1f} std {sym['std P&L']:.1f} "
          f"-> std cut by {1 - inv['std P&L'] / sym['std P&L']:.0%}")
    hi = informed[informed.pi == 0.3].set_index("strategy")
    print("At pi=0.3: " + ", ".join(f"{k}: {v:.1f}" for k, v in hi["mean P&L"].items()))
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
