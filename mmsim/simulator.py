"""Monte Carlo market-making simulator, vectorised over paths.

Time is stepped in increments dt (as in Avellaneda-Stoikov 2008).  Within each
step, Poisson arrivals are approximated by Bernoulli(lambda * dt) events:

    1. the strategy posts a bid and an ask given (t, mid, inventory);
    2. market orders arrive and fill against the quotes at the quoted prices;
    3. the mid moves: S_next = S + sigma sqrt(dt) Z + J * (informed buy - informed sell)
       + public jumps;
    4. the strategy observes the move and its fills (update()).

All randomness is drawn up-front and independently of the quotes, so running
two strategies with the same seed gives them identical price paths and order
arrivals (common random numbers): differences in P&L are due to the strategy only.

P&L decomposition (exact, per step), with n_b / n_a the units bought / sold:

    d(cash + q S) =  n_b (S - bid) + n_a (ask - S)     spread capture
                   + (n_b - n_a)(S_next - S)            fill mark-out (adverse selection)
                   + q_before (S_next - S)              inventory carry
"""

from dataclasses import dataclass

import numpy as np

from .market import MarketParams
from .strategies import Strategy


@dataclass
class SimResult:
    strategy: str
    time: np.ndarray  # (n_steps + 1,)
    mid: np.ndarray  # (n_paths, n_steps + 1)
    bid: np.ndarray  # (n_paths, n_steps)  NaN where not quoting
    ask: np.ndarray  # (n_paths, n_steps)
    inventory: np.ndarray  # (n_paths, n_steps + 1)
    cash: np.ndarray  # (n_paths, n_steps + 1)
    bid_fills: np.ndarray  # (n_paths, n_steps) units bought at our bid
    ask_fills: np.ndarray  # (n_paths, n_steps) units sold at our ask
    informed_fills: np.ndarray  # (n_paths, n_steps) of which were informed
    spread_pnl: np.ndarray  # (n_paths, n_steps) per-step components
    markout_pnl: np.ndarray
    inventory_pnl: np.ndarray

    @property
    def pnl(self) -> np.ndarray:
        """Mark-to-market P&L path: cash + q * mid relative to the start, (n_paths, n_steps + 1)."""
        wealth = self.cash + self.inventory * self.mid
        return wealth - wealth[:, :1]

    @property
    def final_pnl(self) -> np.ndarray:
        return self.pnl[:, -1]

    @property
    def final_inventory(self) -> np.ndarray:
        return self.inventory[:, -1]

    def decomposition(self) -> dict:
        """Mean terminal P&L split into its three components."""
        return {
            "spread capture": self.spread_pnl.sum(axis=1).mean(),
            "adverse selection": self.markout_pnl.sum(axis=1).mean(),
            "inventory carry": self.inventory_pnl.sum(axis=1).mean(),
        }

    def summary(self) -> dict:
        pnl, q = self.final_pnl, self.final_inventory
        spread = (self.ask - self.bid)
        return {
            "strategy": self.strategy,
            "avg spread": np.nanmean(spread),
            "mean P&L": pnl.mean(),
            "std P&L": pnl.std(ddof=1),
            "Sharpe": pnl.mean() / pnl.std(ddof=1),
            "mean final q": q.mean(),
            "std final q": q.std(ddof=1),
            "fills / path": (self.bid_fills + self.ask_fills).sum(axis=1).mean(),
            "informed share": self.informed_fills.sum() / max((self.bid_fills + self.ask_fills).sum(), 1),
        }


def simulate(strategy: Strategy, market: MarketParams, n_paths: int = 1000,
             seed: int = 0, q0: int = 0) -> SimResult:
    m = market
    N, dt = m.n_steps, m.dt
    rng = np.random.default_rng(seed)

    # ---- exogenous randomness (independent of the strategy) ----------------
    z = rng.standard_normal((n_paths, N))
    u_info = rng.random((n_paths, N))  # informed arrival: buyer / seller / none
    u_bid = rng.random((n_paths, N))  # noise seller hitting our bid
    u_ask = rng.random((n_paths, N))  # noise buyer lifting our ask
    u_pub = rng.random((n_paths, N))  # public (non-traded) jumps

    p_inf = m.pi * m.A * dt  # P(informed buyer) = P(informed seller) per step
    informed_buy = u_info < p_inf
    informed_sell = (u_info >= p_inf) & (u_info < 2 * p_inf)
    p_pub = m.public_jump_rate * dt
    public_jump = np.where(u_pub < p_pub / 2, 1.0, np.where(u_pub < p_pub, -1.0, 0.0))

    jumps = m.jump_size * (informed_buy.astype(float) - informed_sell + public_jump)
    dS = m.sigma * np.sqrt(dt) * z + jumps
    mid = m.s0 + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dS, axis=1)], axis=1)

    # ---- state ---------------------------------------------------------------
    shape = (n_paths, N)
    q = np.full(n_paths, q0, dtype=np.int64)
    cash = np.zeros(n_paths)
    out = {name: np.zeros(shape) for name in
           ("bid", "ask", "spread_pnl", "markout_pnl", "inventory_pnl")}
    fills = {name: np.zeros(shape, dtype=np.int64) for name in
             ("bid_fills", "ask_fills", "informed_fills")}
    inv_path = np.zeros((n_paths, N + 1), dtype=np.int64)
    cash_path = np.zeros((n_paths, N + 1))
    inv_path[:, 0] = q

    strategy.reset(n_paths)
    q_max = strategy.max_inventory
    time = np.arange(N + 1) * dt

    for i in range(N):
        s, s_next = mid[:, i], mid[:, i + 1]
        bid, ask = strategy.quote(time[i], s, q)

        # Distance of each quote from mid; no quote = infinitely far = never filled.
        d_bid = np.where(np.isnan(bid), np.inf, s - bid)
        d_ask = np.where(np.isnan(ask), np.inf, ask - s)

        # Noise fills: probability lambda(delta) dt = (1 - pi) A exp(-k delta) dt, capped at 1.
        p_bid = np.minimum((1 - m.pi) * m.A * np.exp(-m.k * d_bid) * dt, 1.0)
        p_ask = np.minimum((1 - m.pi) * m.A * np.exp(-m.k * d_ask) * dt, 1.0)
        noise_b = (u_bid[:, i] < p_bid).astype(np.int64)
        noise_a = (u_ask[:, i] < p_ask).astype(np.int64)

        # Informed fills: they trade with us only if the known jump J beats our quote.
        inf_b = (informed_sell[:, i] & (d_bid < m.jump_size)).astype(np.int64)
        inf_a = (informed_buy[:, i] & (d_ask < m.jump_size)).astype(np.int64)

        # Risk check: reject fills that would breach a hard position limit.
        # Informed traders are fastest, so their orders are matched first.
        if q_max is not None:
            room_b, room_a = np.maximum(q_max - q, 0), np.maximum(q_max + q, 0)
            inf_b = np.minimum(inf_b, room_b)
            noise_b = np.minimum(noise_b, room_b - inf_b)
            inf_a = np.minimum(inf_a, room_a)
            noise_a = np.minimum(noise_a, room_a - inf_a)

        n_b, n_a = noise_b + inf_b, noise_a + inf_a
        bid_px = np.where(n_b > 0, bid, 0.0)  # avoid NaN * 0
        ask_px = np.where(n_a > 0, ask, 0.0)

        # Accounting and P&L attribution.
        out["spread_pnl"][:, i] = n_b * (s - bid_px) + n_a * (ask_px - s)
        out["markout_pnl"][:, i] = (n_b - n_a) * (s_next - s)
        out["inventory_pnl"][:, i] = q * (s_next - s)
        cash = cash - n_b * bid_px + n_a * ask_px
        q = q + n_b - n_a

        out["bid"][:, i], out["ask"][:, i] = bid, ask
        fills["bid_fills"][:, i], fills["ask_fills"][:, i] = n_b, n_a
        fills["informed_fills"][:, i] = inf_b + inf_a
        inv_path[:, i + 1], cash_path[:, i + 1] = q, cash

        strategy.update(s, s_next, n_b, n_a)

    return SimResult(strategy=strategy.name, time=time, mid=mid, inventory=inv_path,
                     cash=cash_path, **out, **fills)
