"""
CommodityPulse Pro — Quant Lab (Phase 2: Scientific Risk & Sizing)
=====================================================================
Shared by app.py (Streamlit terminal) and main.py (background bot).

Honest framing up front: nothing here "ensures maximum profit" — that
doesn't exist. What this module gives you is a more rigorous way to size
positions and to see the *range* of outcomes your edge could plausibly
produce, instead of trusting a single backtest curve at face value.

Contents:
  - kelly_fraction()          Half-Kelly (default) position sizing from
                              realized win rate / avg win / avg loss.
  - time_sliced_stability()   Splits a trade log into contiguous chronological
                              windows to check whether the edge is stable
                              over time. NOTE: this is NOT classic walk-forward
                              optimization (there are no parameters here to
                              re-fit per window) — it's an honest stability /
                              regime-drift check, labeled as such in the UI.
  - monte_carlo_bootstrap()   Bootstrap-resamples the realized trade sequence
                              to build a distribution of alternate equity
                              paths, drawdowns, and a risk-of-ruin estimate.
  - apply_transaction_costs() Net a trade's P&L against MCX brokerage, STT,
                              and slippage assumptions.
  - rolling_correlation()     Pairwise return-correlation matrix from a dict
                              of price series (used to group correlated
                              assets before capping combined risk).
  - correlated_risk_check()   Given currently-open positions and a proposed
                              new one, decide whether combined risk within a
                              correlated group would exceed a cap.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 1. KELLY CRITERION POSITION SIZING
# ─────────────────────────────────────────────────────────────
def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                    kelly_multiplier: float = 0.5, cap: float = 0.10,
                    min_trades: int = 30, n_trades: int | None = None) -> dict:
    """
    Returns the fraction of capital to risk per trade under the Kelly
    Criterion, scaled by kelly_multiplier (0.5 = "Half-Kelly", the standard
    compromise — full Kelly is mathematically optimal for long-run growth
    but has brutal drawdown variance along the way).

    avg_win / avg_loss must be positive numbers (magnitudes, not signed P&L).

    Returns:
        {
          "fraction": float | None,   # e.g. 0.034 = risk 3.4% of capital/trade
          "full_kelly": float | None,
          "payoff_ratio": float | None,
          "usable": bool,             # False if edge is negative or sample too small
          "reason": str,
        }
    """
    if avg_loss is None or avg_loss <= 0 or avg_win is None or avg_win <= 0:
        return {"fraction": None, "full_kelly": None, "payoff_ratio": None,
                "usable": False, "reason": "Insufficient win/loss data to size Kelly."}

    if n_trades is not None and n_trades < min_trades:
        return {"fraction": None, "full_kelly": None, "payoff_ratio": None,
                "usable": False,
                "reason": f"Only {n_trades} trades — need >= {min_trades} for a Kelly estimate that isn't mostly noise."}

    p = max(0.0, min(1.0, win_rate))
    q = 1 - p
    b = avg_win / avg_loss  # payoff ratio

    f_full = (b * p - q) / b
    if f_full <= 0:
        return {"fraction": 0.0, "full_kelly": f_full, "payoff_ratio": b,
                "usable": False,
                "reason": "Full-Kelly is negative or zero — this edge doesn't support Kelly sizing. Falling back to fixed %."}

    f_scaled = min(f_full * kelly_multiplier, cap)
    return {"fraction": f_scaled, "full_kelly": f_full, "payoff_ratio": b,
            "usable": True,
            "reason": f"Half-Kelly (x{kelly_multiplier}) of a {f_full*100:.1f}% full-Kelly edge, capped at {cap*100:.0f}%."}


# ─────────────────────────────────────────────────────────────
# 2. TIME-SLICED STABILITY CHECK (simplified walk-forward)
# ─────────────────────────────────────────────────────────────
def time_sliced_stability(trade_df: pd.DataFrame, n_windows: int = 4,
                           time_col: str = "time", pnl_col: str = "pnl") -> list[dict]:
    """
    Splits a chronologically-sorted trade log into n_windows contiguous
    slices and computes per-window stats. This checks whether an edge
    holds up over time or was concentrated in one lucky stretch — it does
    NOT re-fit any parameters per window (there aren't tunable parameters
    in these rule-based strategies), so it is not classic walk-forward
    optimization. Think of it as a "does this edge persist" check.
    """
    if trade_df is None or trade_df.empty or n_windows < 2:
        return []

    d = trade_df.sort_values(time_col).reset_index(drop=True)
    n = len(d)
    if n < n_windows * 5:
        return []  # too few trades per window to mean anything

    edges = np.linspace(0, n, n_windows + 1).astype(int)
    windows = []
    for i in range(n_windows):
        chunk = d.iloc[edges[i]:edges[i+1]]
        if chunk.empty:
            continue
        wins = chunk[chunk[pnl_col] > 0]
        losses = chunk[chunk[pnl_col] <= 0]
        gross_profit = wins[pnl_col].sum()
        gross_loss = -losses[pnl_col].sum()
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        windows.append({
            "window": i + 1,
            "start": chunk[time_col].iloc[0],
            "end": chunk[time_col].iloc[-1],
            "trades": len(chunk),
            "win_rate": len(wins) / len(chunk) * 100,
            "profit_factor": pf,
            "expectancy": chunk[pnl_col].mean(),
            "net_pnl": chunk[pnl_col].sum(),
        })
    return windows


# ─────────────────────────────────────────────────────────────
# 3. MONTE CARLO BOOTSTRAP / RISK OF RUIN
# ─────────────────────────────────────────────────────────────
def monte_carlo_bootstrap(trade_pnls: pd.Series | np.ndarray, start_equity: float,
                           n_sims: int = 2000, n_trades: int | None = None,
                           ruin_dd_pct: float = 40.0, seed: int = 42) -> dict:
    """
    Bootstrap-resamples the realized trade P&L sequence (with replacement)
    to build a distribution of alternate equity paths. Your actual backtest
    is just ONE possible ordering of these trades — this shows the range of
    plausible outcomes if the same edge played out in a different order or
    got unlucky/lucky streaks clustered differently.

    Returns percentile equity curves (5/25/50/75/95), and headline risk
    metrics: probability of ending profitable, and a "risk of ruin" estimate
    (probability that max drawdown exceeds ruin_dd_pct at any point).
    """
    pnls = np.asarray(trade_pnls, dtype=float)
    pnls = pnls[~np.isnan(pnls)]
    if len(pnls) < 10:
        return {"usable": False, "reason": "Need at least 10 completed trades for a meaningful simulation."}

    rng = np.random.default_rng(seed)
    n_trades = n_trades or len(pnls)

    sims = rng.choice(pnls, size=(n_sims, n_trades), replace=True)
    equity_paths = start_equity + np.cumsum(sims, axis=1)
    equity_paths = np.hstack([np.full((n_sims, 1), start_equity), equity_paths])

    running_max = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns_pct = (equity_paths - running_max) / running_max * 100
    max_dd_per_sim = drawdowns_pct.min(axis=1)

    final_equity = equity_paths[:, -1]
    prob_profit = float((final_equity > start_equity).mean() * 100)
    prob_ruin = float((max_dd_per_sim <= -abs(ruin_dd_pct)).mean() * 100)

    pct_levels = [5, 25, 50, 75, 95]
    percentile_curves = {p: np.percentile(equity_paths, p, axis=0) for p in pct_levels}

    return {
        "usable": True,
        "n_sims": n_sims, "n_trades": n_trades, "start_equity": start_equity,
        "percentile_curves": percentile_curves,
        "prob_profit_pct": prob_profit,
        "prob_ruin_pct": prob_ruin,
        "ruin_dd_pct": ruin_dd_pct,
        "median_final_equity": float(np.median(final_equity)),
        "p5_final_equity": float(np.percentile(final_equity, 5)),
        "p95_final_equity": float(np.percentile(final_equity, 95)),
        "median_max_dd_pct": float(np.median(max_dd_per_sim)),
        "worst_5pct_dd_pct": float(np.percentile(max_dd_per_sim, 5)),
    }


# ─────────────────────────────────────────────────────────────
# 4. TRANSACTION COSTS (brokerage + STT + slippage)
# ─────────────────────────────────────────────────────────────
def apply_transaction_costs(gross_pnl: float, entry_price: float, lots: int, lot_size: float,
                             brokerage_per_lot: float = 20.0, stt_rate: float = 0.0001,
                             slippage_pct: float = 0.0005) -> dict:
    """
    Nets a single trade's gross P&L against estimated round-trip costs:
      - Flat brokerage per lot, charged on entry AND exit (x2)
      - STT (Securities Transaction Tax) as a % of notional turnover
      - Slippage as a % of entry price, applied to the traded notional

    These are editable assumptions, not MCX-official figures — actual costs
    vary by broker and contract. Treat this as "does the edge survive
    realistic friction," not a precise cost simulator.
    """
    notional = entry_price * lots * lot_size
    brokerage_cost = brokerage_per_lot * lots * 2         # entry + exit
    stt_cost       = notional * stt_rate
    slippage_cost  = notional * slippage_pct
    total_cost     = brokerage_cost + stt_cost + slippage_cost
    return {
        "gross_pnl": gross_pnl,
        "net_pnl": gross_pnl - total_cost,
        "total_cost": total_cost,
        "brokerage_cost": brokerage_cost,
        "stt_cost": stt_cost,
        "slippage_cost": slippage_cost,
    }


# ─────────────────────────────────────────────────────────────
# 5. CORRELATION & PORTFOLIO RISK CAP
# ─────────────────────────────────────────────────────────────
def rolling_correlation(price_series: dict[str, pd.Series], lookback: int = 60) -> pd.DataFrame:
    """
    Pairwise correlation matrix of daily returns across tickers, over the
    last `lookback` observations. price_series: {ticker: pd.Series of closes}.
    """
    returns = {}
    for ticker, s in price_series.items():
        s = s.dropna()
        if len(s) < lookback + 1:
            continue
        returns[ticker] = s.pct_change().dropna().iloc[-lookback:]
    if len(returns) < 2:
        return pd.DataFrame()
    ret_df = pd.DataFrame(returns).dropna()
    if ret_df.empty:
        return pd.DataFrame()
    return ret_df.corr()


def correlated_risk_check(ticker: str, direction: str, new_risk_inr: float,
                           open_positions: dict, corr_matrix: pd.DataFrame,
                           capital_inr: float, max_group_risk_pct: float = 4.0,
                           corr_threshold: float = 0.6) -> dict:
    """
    open_positions: {ticker: {"direction": "BULLISH"|"BEARISH", "risk_inr": float, ...}}
    corr_matrix: output of rolling_correlation()

    Sums risk already committed to positions that are BOTH correlated with
    `ticker` (|corr| >= corr_threshold) AND in the same effective direction
    (same direction if corr > 0, opposite direction if corr < 0 — since a
    negatively-correlated pair moving together in the SAME nominal direction
    is actually a hedge, not a stacked bet). If adding new_risk_inr would push
    that group's combined risk over max_group_risk_pct of capital, the new
    position is flagged to skip.
    """
    if corr_matrix is None or corr_matrix.empty or ticker not in corr_matrix.columns:
        return {"allowed": True, "grouped_risk_inr": 0.0, "reason": "No correlation data available — allowing by default."}

    grouped_risk = 0.0
    grouped_with = []
    for other_ticker, pos in open_positions.items():
        if other_ticker == ticker or other_ticker not in corr_matrix.columns:
            continue
        corr = corr_matrix.loc[ticker, other_ticker]
        if pd.isna(corr) or abs(corr) < corr_threshold:
            continue
        same_side = (pos["direction"] == direction) if corr > 0 else (pos["direction"] != direction)
        if same_side:
            grouped_risk += pos.get("risk_inr", 0.0)
            grouped_with.append(f"{other_ticker} (corr={corr:+.2f})")

    prospective_total = grouped_risk + new_risk_inr
    cap_inr = capital_inr * (max_group_risk_pct / 100.0)
    allowed = prospective_total <= cap_inr

    reason = (
        f"Combined correlated risk would be ₹{prospective_total:,.0f} "
        f"({prospective_total/capital_inr*100:.2f}% of capital) vs cap of "
        f"{max_group_risk_pct:.1f}% (₹{cap_inr:,.0f})."
    )
    if grouped_with:
        reason += f" Correlated with open: {', '.join(grouped_with)}."

    return {"allowed": allowed, "grouped_risk_inr": grouped_risk,
            "prospective_total_inr": prospective_total, "cap_inr": cap_inr, "reason": reason}
