"""Market parameters and the analytical results that go with them.

The market (one stock, one market maker) is described by:

* the mid-price  S_t, an arithmetic Brownian motion  dS = sigma dW  plus jumps;
* uninformed ("noise") market orders, arriving on each side at rate (1 - pi) * A.
  A noise order trades against our quote at distance delta from mid with
  probability exp(-k * delta), so our fill intensity is
  lambda(delta) = (1 - pi) * A * exp(-k * delta)      (Avellaneda-Stoikov 2008);
* informed market orders, arriving on each side at rate pi * A.  An informed
  trader knows that the mid is about to jump by +J (buyer) or -J (seller) and
  trades with us only if that is profitable, i.e. if our quote is closer than J
  to the mid (Glosten-Milgrom 1985 flavour).  The jump happens whether or not
  they trade with us: the news becomes public.

With pi = 0 we recover exactly the Avellaneda-Stoikov market.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class MarketParams:
    s0: float = 100.0  # initial mid-price
    T: float = 1.0  # horizon
    sigma: float = 2.0  # diffusion volatility of the mid (price units / sqrt(time))
    dt: float = 0.005  # time step
    A: float = 140.0  # market order arrival rate per side
    k: float = 1.5  # decay of fill probability with distance from mid
    pi: float = 0.0  # fraction of market orders that are informed
    jump_size: float = 1.5  # J: size of the price move an informed trader knows about
    public_jump_rate: float = 0.0  # optional extra +/-J jumps per unit time nobody trades on

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    def __post_init__(self):
        if not 0.0 <= self.pi <= 1.0:
            raise ValueError("pi must be in [0, 1]")
        # Each side sees at most one noise and one informed arrival per step, and
        # informed buyer/seller arrivals are drawn as mutually exclusive events,
        # so these probabilities must be valid.
        if self.A * self.dt > 1.0 or 2 * self.pi * self.A * self.dt > 1.0:
            raise ValueError("dt too large: arrival probabilities per step exceed 1")
        if self.public_jump_rate * self.dt > 1.0:
            raise ValueError("dt too large for public_jump_rate")


def expected_pnl_rate(delta, m: MarketParams):
    """Expected P&L per unit time of a symmetric quoter at half-spread delta.

    Per side: noise fills earn delta, informed fills earn delta - J.
        (1 - pi) A e^{-k delta} * delta  -  pi A * 1{delta < J} * (J - delta)
    Inventory P&L has zero mean because the mid is a martingale, so this is
    exact (also in the discrete-time simulator).  Factor 2 for two sides.
    """
    delta = np.asarray(delta, dtype=float)
    noise = (1 - m.pi) * m.A * np.exp(-m.k * delta) * delta
    informed = m.pi * m.A * np.where(delta < m.jump_size, m.jump_size - delta, 0.0)
    return 2.0 * (noise - informed)


def break_even_half_spread(m: MarketParams) -> float:
    """Smallest half-spread at which a symmetric quoter does not lose money.

    Solves (1 - pi) e^{-k d} d = pi (J - d) for d in [0, J].  For pi = 0 any
    positive spread is profitable (returns 0); as pi -> 1 it tends to J.
    With inelastic noise demand (k = 0) this is the Glosten-Milgrom answer d = pi J.
    """
    if m.pi == 0.0:
        return 0.0
    if m.pi == 1.0:
        return m.jump_size

    def f(d):
        return (1 - m.pi) * np.exp(-m.k * d) * d - m.pi * (m.jump_size - d)

    # f(0) < 0 < f(J). Bracket the first sign change on a grid, then refine.
    grid = np.linspace(0.0, m.jump_size, 1001)
    first = np.argmax(f(grid) > 0)
    return brentq(f, grid[first - 1], grid[first])
