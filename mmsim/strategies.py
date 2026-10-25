"""Quoting strategies.

Every strategy implements the same small interface, vectorised over Monte Carlo
paths (all arrays have shape (n_paths,)):

    reset(n_paths)                      -> clear any per-path state
    quote(t, mid, inventory)            -> (bid, ask) prices; NaN = no quote
    update(mid_before, mid_after,
           bid_fills, ask_fills)        -> learn from what just happened

`max_inventory`, if set, is a hard position limit that the simulator's risk
check also enforces.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov (2008) closed-form formulas
# ---------------------------------------------------------------------------

def reservation_price(mid, inventory, gamma, sigma, time_left):
    """r = s - q * gamma * sigma^2 * (T - t).

    The price at which the market maker is indifferent to holding one more unit:
    long inventory (q > 0) pushes r below mid, so we quote lower to sell.
    """
    return mid - inventory * gamma * sigma**2 * time_left


def optimal_spread(gamma, sigma, k, time_left):
    """Total optimal bid-ask spread  gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k).

    First term: compensation for inventory risk over the remaining horizon.
    Second term: the monopoly mark-up given how fast fills decay with distance (k).
    """
    return gamma * sigma**2 * time_left + (2.0 / gamma) * np.log1p(gamma / k)


def average_as_spread(gamma, sigma, k, T, dt):
    """Time-average of the A-S spread over the simulation grid t = 0, dt, ..., T - dt.

    Used to calibrate the symmetric benchmark exactly as in the paper.
    """
    time_left = T - np.arange(int(round(T / dt))) * dt
    return float(np.mean(optimal_spread(gamma, sigma, k, time_left)))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class Strategy:
    name = "strategy"
    max_inventory = None

    def reset(self, n_paths: int) -> None:
        pass

    def quote(self, t, mid, inventory):
        raise NotImplementedError

    def update(self, mid_before, mid_after, bid_fills, ask_fills) -> None:
        pass


class SymmetricQuoter(Strategy):
    """Naive benchmark: quote mid +/- a fixed half-spread, ignoring inventory."""

    def __init__(self, half_spread: float, name: str = "Symmetric"):
        self.half_spread = half_spread
        self.name = name

    def quote(self, t, mid, inventory):
        return mid - self.half_spread, mid + self.half_spread


class AvellanedaStoikov(Strategy):
    """Quote symmetrically around the reservation price r with the optimal spread."""

    def __init__(self, gamma, sigma, k, T, name="Avellaneda-Stoikov"):
        self.gamma, self.sigma, self.k, self.T = gamma, sigma, k, T
        self.name = name

    def quote(self, t, mid, inventory):
        time_left = self.T - t
        r = reservation_price(mid, inventory, self.gamma, self.sigma, time_left)
        half = 0.5 * optimal_spread(self.gamma, self.sigma, self.k, time_left)
        return r - half, r + half


class AvellanedaStoikovLimits(AvellanedaStoikov):
    """A-S quotes, but stop quoting the side that would push |q| past max_inventory."""

    def __init__(self, gamma, sigma, k, T, max_inventory: int, name="A-S + limits"):
        super().__init__(gamma, sigma, k, T, name=name)
        self.max_inventory = max_inventory

    def quote(self, t, mid, inventory):
        bid, ask = super().quote(t, mid, inventory)
        bid = np.where(inventory >= self.max_inventory, np.nan, bid)
        ask = np.where(inventory <= -self.max_inventory, np.nan, ask)
        return bid, ask


class AdaptiveToxicityQuoter(AvellanedaStoikovLimits):
    """A-S with limits, plus a per-side toxicity premium learned online from mark-outs.

    After every fill we observe the next mid move.  The *mark-out loss* of a fill
    is how much the price moved against us:
        sold at the ask  -> loss = +(S_next - S)     (price went up after we sold)
        bought at bid    -> loss = -(S_next - S)     (price went down after we bought)
    Noise fills have zero expected loss; informed fills lose J.  An exponentially
    weighted average of these losses, kept separately for each side, estimates the
    expected adverse-selection cost per fill.  We add it to that side's distance
    from the reservation price, so a run of buyers lifting our ask followed by the
    price rising (one-sided toxic flow) widens the ask, and the premium decays
    back towards zero once fills stop losing money.  No knowledge of pi or J is used.
    """

    def __init__(self, gamma, sigma, k, T, max_inventory, alpha=0.1, name="Adaptive (toxicity)"):
        super().__init__(gamma, sigma, k, T, max_inventory, name=name)
        self.alpha = alpha  # EWMA weight per fill: memory of roughly 1 / alpha fills

    def reset(self, n_paths):
        self.tox_bid = np.zeros(n_paths)
        self.tox_ask = np.zeros(n_paths)

    def quote(self, t, mid, inventory):
        bid, ask = super().quote(t, mid, inventory)
        return bid - np.maximum(self.tox_bid, 0.0), ask + np.maximum(self.tox_ask, 0.0)

    def update(self, mid_before, mid_after, bid_fills, ask_fills):
        move = mid_after - mid_before
        hit = bid_fills > 0
        self.tox_bid[hit] += self.alpha * (-move[hit] - self.tox_bid[hit])
        hit = ask_fills > 0
        self.tox_ask[hit] += self.alpha * (move[hit] - self.tox_ask[hit])
