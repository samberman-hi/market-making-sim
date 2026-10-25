import numpy as np
import pytest

from mmsim import (
    AdaptiveToxicityQuoter,
    AvellanedaStoikov,
    AvellanedaStoikovLimits,
    MarketParams,
    SymmetricQuoter,
    expected_pnl_rate,
    simulate,
)

TOXIC = MarketParams(pi=0.2, public_jump_rate=5.0)


def all_strategies(max_inventory=5):
    return [
        SymmetricQuoter(0.7),
        AvellanedaStoikov(0.1, 2.0, 1.5, 1.0),
        AvellanedaStoikovLimits(0.1, 2.0, 1.5, 1.0, max_inventory),
        AdaptiveToxicityQuoter(0.1, 2.0, 1.5, 1.0, max_inventory),
    ]


@pytest.mark.parametrize("strategy", all_strategies(), ids=lambda s: s.name)
def test_accounting_identity_every_step(strategy):
    """cash + q * S - initial wealth == cumulative (spread + mark-out + inventory) P&L."""
    res = simulate(strategy, TOXIC, n_paths=300, seed=7, q0=3)
    components = res.spread_pnl + res.markout_pnl + res.inventory_pnl
    cumulative = np.concatenate([np.zeros((300, 1)), np.cumsum(components, axis=1)], axis=1)
    assert np.allclose(res.pnl, cumulative, atol=1e-9)

    # Inventory only changes through fills; cash only through trades at quoted prices.
    dq = np.diff(res.inventory, axis=1)
    assert np.array_equal(dq, res.bid_fills - res.ask_fills)
    bid_px = np.where(res.bid_fills > 0, res.bid, 0.0)
    ask_px = np.where(res.ask_fills > 0, res.ask, 0.0)
    assert np.allclose(np.diff(res.cash, axis=1), res.ask_fills * ask_px - res.bid_fills * bid_px)


@pytest.mark.parametrize("delta", [0.2, 0.8])
def test_noise_fill_probability_matches_intensity(delta):
    """Empirical fill frequency per step == lambda(delta) dt = A exp(-k delta) dt."""
    m = MarketParams()
    res = simulate(SymmetricQuoter(delta), m, n_paths=5000, seed=1)
    p = m.A * np.exp(-m.k * delta) * m.dt
    se = np.sqrt(p * (1 - p) / res.bid_fills.size)
    assert res.bid_fills.mean() == pytest.approx(p, abs=4 * se)
    assert res.ask_fills.mean() == pytest.approx(p, abs=4 * se)


def test_informed_traders_only_pick_off_quotes_inside_the_jump():
    m = MarketParams(pi=0.3, jump_size=1.0)
    p_inf = m.pi * m.A * m.dt
    tight = simulate(SymmetricQuoter(0.5), m, n_paths=3000, seed=2)
    # Both sides are inside J: P(informed fill in a step) = P(buyer) + P(seller) = 2 p_inf
    se = np.sqrt(2 * p_inf * (1 - 2 * p_inf) / tight.informed_fills.size)
    assert tight.informed_fills.mean() == pytest.approx(2 * p_inf, abs=4 * se)
    wide = simulate(SymmetricQuoter(1.2), m, n_paths=500, seed=2)
    assert wide.informed_fills.sum() == 0


def test_symmetric_quoter_profitable_without_risk_or_information():
    """pi = 0 and sigma = 0: no inventory risk, P&L is pure spread capture > 0."""
    m = MarketParams(sigma=0.0, pi=0.0)
    delta = 0.7
    res = simulate(SymmetricQuoter(delta), m, n_paths=2000, seed=3)
    assert np.all(res.final_pnl >= 0)
    expected = expected_pnl_rate(delta, m) * m.T  # 2 A e^{-k delta} delta T
    se = res.final_pnl.std(ddof=1) / np.sqrt(2000)
    assert res.final_pnl.mean() > 0
    assert res.final_pnl.mean() == pytest.approx(expected, abs=4 * se)


def test_expected_pnl_matches_theory_with_informed_traders():
    """With inventory risk and adverse selection the mean P&L still matches theory."""
    for pi, delta in [(0.1, 0.4), (0.3, 0.7)]:
        m = MarketParams(pi=pi)
        res = simulate(SymmetricQuoter(delta), m, n_paths=4000, seed=4)
        se = res.final_pnl.std(ddof=1) / np.sqrt(4000)
        assert res.final_pnl.mean() == pytest.approx(expected_pnl_rate(delta, m) * m.T, abs=4 * se)


@pytest.mark.parametrize("strategy", all_strategies(max_inventory=2)[2:], ids=lambda s: s.name)
def test_inventory_limits_respected(strategy):
    m = MarketParams(pi=0.4, A=190.0)  # very active market: limits bind often
    res = simulate(strategy, m, n_paths=500, seed=5)
    assert np.abs(res.inventory).max() <= 2
    assert np.abs(res.inventory).max() == 2  # and they actually bind


def test_risk_check_enforces_limit_for_any_strategy():
    naive = SymmetricQuoter(0.0)  # quotes at mid: fills almost every step
    naive.max_inventory = 1
    res = simulate(naive, MarketParams(pi=0.3), n_paths=500, seed=6)
    assert np.abs(res.inventory).max() <= 1


def test_common_random_numbers_and_reproducibility():
    a = simulate(SymmetricQuoter(0.5), TOXIC, n_paths=50, seed=11)
    b = simulate(AvellanedaStoikov(0.1, 2.0, 1.5, 1.0), TOXIC, n_paths=50, seed=11)
    c = simulate(SymmetricQuoter(0.5), TOXIC, n_paths=50, seed=11)
    assert np.array_equal(a.mid, b.mid)  # same market for every strategy
    assert np.array_equal(a.final_pnl, c.final_pnl)  # same seed, same result


def test_adaptive_quoter_learns_toxicity():
    strat = AdaptiveToxicityQuoter(0.1, 2.0, 1.5, 1.0, max_inventory=10)
    simulate(strat, MarketParams(pi=0.0), n_paths=500, seed=8)
    assert abs(strat.tox_ask.mean()) < 0.05  # no informed flow: premium ~ 0
    toxic = simulate(strat, MarketParams(pi=0.3), n_paths=500, seed=8)
    assert strat.tox_ask.mean() > 0.3 and strat.tox_bid.mean() > 0.3
    plain = simulate(AvellanedaStoikovLimits(0.1, 2.0, 1.5, 1.0, 10), MarketParams(pi=0.3),
                     n_paths=500, seed=8)
    assert toxic.final_pnl.mean() > plain.final_pnl.mean()
