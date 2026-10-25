import numpy as np
import pytest

from mmsim import (
    AvellanedaStoikov,
    AvellanedaStoikovLimits,
    MarketParams,
    average_as_spread,
    break_even_half_spread,
    expected_pnl_rate,
    optimal_spread,
    reservation_price,
)


def test_reservation_price_hand_computed():
    # r = s - q * gamma * sigma^2 * (T - t) = 100 - 2 * 0.1 * 4 * 0.5 = 99.6
    assert reservation_price(100.0, 2, 0.1, 2.0, 0.5) == pytest.approx(99.6)
    # Short 3 units, gamma = 0.5, T - t = 1: 100 + 3 * 0.5 * 4 = 106
    assert reservation_price(100.0, -3, 0.5, 2.0, 1.0) == pytest.approx(106.0)


def test_optimal_spread_hand_computed():
    # gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma/k)
    # = 0.1 * 4 * 0.5 + 20 * ln(1 + 0.1/1.5) = 0.2 + 20 * 0.0645385211 = 1.490770
    assert optimal_spread(0.1, 2.0, 1.5, 0.5) == pytest.approx(1.490770, abs=1e-6)
    # At expiry only the "monopoly" term is left: 4 * ln(4/3) = 1.150728
    assert optimal_spread(0.5, 2.0, 1.5, 0.0) == pytest.approx(1.150728, abs=1e-6)
    # As gamma -> 0 the spread tends to the risk-neutral 2/k
    assert optimal_spread(1e-8, 2.0, 1.5, 1.0) == pytest.approx(2 / 1.5, rel=1e-6)


def test_average_spread_matches_paper_constant_term():
    # The paper's table reports (2/gamma) ln(1 + gamma/k); the average spread adds
    # gamma sigma^2 * mean(T - t) = gamma * 4 * (1 + dt) / 2 on our grid.
    for gamma, paper in [(0.01, 1.33), (0.1, 1.29), (0.5, 1.15)]:
        avg = average_as_spread(gamma, 2.0, 1.5, 1.0, 0.005)
        assert avg - gamma * 4 * (1 + 0.005) / 2 == pytest.approx(paper, abs=0.005)


def test_reservation_price_skews_against_inventory():
    strat = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    mid = np.full(3, 100.0)
    q = np.array([-5, 0, 5])
    bid, ask = strat.quote(0.3, mid, q)
    centre = 0.5 * (bid + ask)
    assert centre[0] > 100.0  # short -> quote higher to buy back
    assert centre[1] == pytest.approx(100.0)  # flat -> symmetric around mid
    assert centre[2] < 100.0  # long -> quote lower to sell
    # The spread itself does not depend on inventory
    assert np.allclose(ask - bid, optimal_spread(0.1, 2.0, 1.5, 0.7))


def test_limit_strategy_pulls_quote_at_limit():
    strat = AvellanedaStoikovLimits(0.1, 2.0, 1.5, 1.0, max_inventory=3)
    bid, ask = strat.quote(0.0, np.full(3, 100.0), np.array([3, 0, -3]))
    assert np.isnan(bid[0]) and not np.isnan(ask[0])
    assert not np.isnan(bid[1]) and not np.isnan(ask[1])
    assert np.isnan(ask[2]) and not np.isnan(bid[2])


def test_break_even_spread():
    # Glosten-Milgrom limit: with inelastic noise demand (k = 0) break-even is pi * J
    m = MarketParams(k=0.0, pi=0.3, jump_size=1.5)
    assert break_even_half_spread(m) == pytest.approx(0.45, abs=1e-9)
    # In general expected P&L is exactly zero at the break-even spread, negative below
    m = MarketParams(pi=0.2)
    d = break_even_half_spread(m)
    assert expected_pnl_rate(d, m) == pytest.approx(0.0, abs=1e-8)
    assert expected_pnl_rate(0.9 * d, m) < 0 < expected_pnl_rate(1.1 * d, m)
    assert break_even_half_spread(MarketParams(pi=0.0)) == 0.0
