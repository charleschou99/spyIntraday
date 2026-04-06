"""
Idiosyncratic Volatility Dispersion Trading Module

Based on Assenagon's approach to idiosyncratic volatility decomposition:
1. Decompose stock returns into systematic (beta) and idiosyncratic (residual) components
2. Trade dispersion when implied correlation diverges from realized correlation
3. Use idiosyncratic volatility as a timing signal

Required input format
---------------------
Index (daily or intraday):
- pandas DatetimeIndex, timezone-aware
- Columns: market_return, index_implied_vol

Constituents:
- Dictionary or DataFrame with columns:
    - stock_return (for each constituent)
    - stock_implied_vol (for each constituent)
    - weight (index weight for each constituent)

Linear Regression Equations
---------------------------
1. Return decomposition:
   r_i,t = alpha_i + beta_i * r_m,t + epsilon_i,t

2. Idiosyncratic volatility:
   sigma_idio,i = sqrt(Var(epsilon_i))

3. Index variance decomposition:
   sigma_m^2 = sum_i w_i^2 * sigma_i^2 + sum_i!=j w_i * w_j * rho_ij * sigma_i * sigma_j

4. Implied correlation:
   rho_implied = (sigma_index,impl^2 - sum_i w_i^2 * sigma_i,impl^2) / sum_i!=j w_i*w_j*sigma_i*sigma_j

5. Dispersion trading signal:
   Dispersion_spread = sigma_index,impl - sum_i w_i * sigma_i,impl

6. Regression for trade prediction:
   Realized_Dispersion = gamma_0 + gamma_1 * Implied_Dispersion + gamma_2 * IV_Signal + eta
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import lstsq


@dataclass(frozen=True)
class DispersionParams:
    """Parameters for dispersion trading strategy."""
    lookback_window: int = 21  # Days for rolling regression/calculation
    signal_threshold: float = 1.5  # Z-score threshold for trade entry
    trade_mode: str = "short_dispersion"  # "short_dispersion" or "long_dispersion"
    # When to enter: when implied correlation is rich (short) or cheap (long)
    use_idio_vol_filter: bool = True
    idio_vol_percentile_threshold: float = 75.0  # Trade when idio vol is high


@dataclass
class RegressionResult:
    """Stores the results of return decomposition regression."""
    stock_symbol: str
    alpha: float  # Intercept (Jensen's alpha)
    beta: float  # Market beta
    r_squared: float  # Goodness of fit
    idio_volatility: float  # Standard deviation of residuals (key metric!)
    systematic_volatility: float  # Beta * market_vol
    total_volatility: float  # Total realized volatility
    residuals: pd.Series  # Idiosyncratic returns (epsilon)


def decompose_returns(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    stock_symbol: str = "STOCK",
) -> RegressionResult:
    """
    Decompose stock returns using linear regression:
    
    r_i = alpha + beta * r_m + epsilon
    
    Where epsilon represents the idiosyncratic (stock-specific) component.
    
    Args:
        stock_returns: Series of stock returns
        market_returns: Series of market/index returns
        stock_symbol: Identifier for the stock
        
    Returns:
        RegressionResult with alpha, beta, and idiosyncratic volatility
    """
    # Align and clean data
    data = pd.DataFrame({"stock": stock_returns, "market": market_returns}).dropna()
    
    if len(data) < 10:
        raise ValueError(f"Insufficient data points ({len(data)}) for regression")
    
    # Linear regression: r_stock = alpha + beta * r_market + epsilon
    X = np.column_stack([np.ones(len(data)), data["market"].values])  # Add intercept
    y = data["stock"].values
    
    # Solve for [alpha, beta]
    coeffs, residuals, rank, s = lstsq(X, y, rcond=None)
    alpha, beta = coeffs[0], coeffs[1]
    
    # Calculate residuals (idiosyncratic returns)
    fitted = X @ coeffs
    epsilon = y - fitted
    
    # Volatility metrics
    total_var = np.var(y, ddof=1)
    idio_var = np.var(epsilon, ddof=1)
    systematic_var = np.var(fitted, ddof=1)
    
    total_vol = np.sqrt(total_var)
    idio_vol = np.sqrt(idio_var)
    systematic_vol = np.sqrt(systematic_var)
    
    # R-squared
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(epsilon ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Store residuals as series
    epsilon_series = pd.Series(epsilon, index=data.index, name="residuals")
    
    return RegressionResult(
        stock_symbol=stock_symbol,
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        idio_volatility=idio_vol,
        systematic_volatility=systematic_vol,
        total_volatility=total_vol,
        residuals=epsilon_series,
    )


def compute_index_idiosyncratic_ratio(
    constituents_returns: pd.DataFrame,
    market_returns: pd.Series,
    weights: pd.Series,
    lookback: int = 21,
) -> pd.Series:
    """
    Compute the aggregate idiosyncratic volatility ratio for an index.
    
    IV_Signal_t = (sum_i w_i * sigma_idio,i,t) / sigma_m,t
    
    This ratio measures how much of total market volatility is stock-specific
    versus systematic. High values = high dispersion opportunity.
    
    Args:
        constituents_returns: DataFrame with each stock's returns as columns
        market_returns: Series of market/index returns
        weights: Series of index weights (same order as columns)
        lookback: Rolling window for volatility calculation
        
    Returns:
        Series of idiosyncratic-to-total volatility ratios
    """
    if len(weights) != len(constituents_returns.columns):
        raise ValueError("Number of weights must match number of constituents")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Compute idiosyncratic vol for each stock over rolling window
    idio_vols = {}
    
    for col in constituents_returns.columns:
        stock_rets = constituents_returns[col]
        
        # Rolling regression to get rolling idio vol
        rolling_idio = stock_rets.rolling(window=lookback).apply(
            lambda x: _rolling_idio_vol(x, market_returns.loc[x.index]),
            raw=False,
        )
        idio_vols[col] = rolling_idio
    
    idio_df = pd.DataFrame(idio_vols)
    
    # Weighted average of idiosyncratic volatilities
    weighted_idio = (idio_df * weights.values).sum(axis=1)
    
    # Market volatility
    market_vol = market_returns.rolling(window=lookback).std() * np.sqrt(252)
    
    # Idiosyncratic ratio
    idio_ratio = weighted_idio / market_vol
    
    return idio_ratio


def _rolling_idio_vol(stock_rets: pd.Series, market_rets: pd.Series) -> float:
    """Helper function to compute idiosyncratic volatility for a rolling window."""
    if len(stock_rets) < 10:
        return np.nan
    
    try:
        result = decompose_returns(stock_rets, market_rets)
        return result.idio_volatility * np.sqrt(252)  # Annualized
    except:
        return np.nan


def compute_implied_dispersion(
    index_implied_vol: float,
    constituent_implied_vols: pd.Series,
    weights: pd.Series,
) -> Dict[str, float]:
    """
    Calculate implied dispersion metrics.
    
    Key equation:
    Dispersion_spread = sigma_index,impl - sum_i (w_i * sigma_i,impl)
    
    Args:
        index_implied_vol: Implied volatility of the index (e.g., SPX)
        constituent_implied_vols: Implied vols of constituents
        weights: Index weights for each constituent
        
    Returns:
        Dictionary with dispersion metrics
    """
    # Normalize weights
    weights = weights / weights.sum()
    
    # Weighted average of constituent implied vols
    weighted_avg_iv = (constituent_implied_vols * weights).sum()
    
    # Dispersion spread
    dispersion_spread = index_implied_vol - weighted_avg_iv
    
    # Implied correlation (simplified)
    # rho_impl = (sigma_index^2 - sum(w_i^2 * sigma_i^2)) / (sum_i!=j w_i*w_j*sigma_i*sigma_j)
    sum_squared_w_vols = ((weights ** 2) * (constituent_implied_vols ** 2)).sum()
    
    # Approximate: assume equal cross-volatilities
    n = len(weights)
    avg_weight = 1 / n
    cross_term_denom = (1 - (weights ** 2).sum()) * (weighted_avg_iv ** 2)
    
    numerator = index_implied_vol ** 2 - sum_squared_w_vols
    implied_correlation = numerator / cross_term_denom if cross_term_denom > 0 else 0
    
    return {
        "index_iv": index_implied_vol,
        "weighted_constituent_iv": weighted_avg_iv,
        "dispersion_spread": dispersion_spread,
        "implied_correlation": np.clip(implied_correlation, -1, 1),
    }


def compute_realized_dispersion(
    constituent_returns: pd.DataFrame,
    index_returns: pd.Series,
    weights: pd.Series,
    lookback: int = 21,
) -> Dict[str, float]:
    """
    Calculate realized dispersion over a lookback period.
    
    Realized dispersion = sigma_index,realized - sum_i (w_i * sigma_i,realized)
    
    This is what we use to validate our trades.
    """
    weights = weights / weights.sum()
    
    # Realized volatilities (annualized)
    stock_vols = constituent_returns.rolling(window=lookback).std() * np.sqrt(252)
    index_vol = index_returns.rolling(window=lookback).std() * np.sqrt(252)
    
    # Latest values
    latest_stock_vols = stock_vols.iloc[-1]
    latest_index_vol = index_vol.iloc[-1]
    
    weighted_stock_vol = (latest_stock_vols * weights).sum()
    realized_dispersion = latest_index_vol - weighted_stock_vol
    
    return {
        "index_realized_vol": latest_index_vol,
        "weighted_stock_realized_vol": weighted_stock_vol,
        "realized_dispersion": realized_dispersion,
    }


def build_dispersion_signal(
    index_implied_vol_history: pd.Series,
    constituent_implied_vols_history: pd.DataFrame,
    constituent_returns_history: pd.DataFrame,
    index_returns_history: pd.Series,
    weights: pd.Series,
    lookback: int = 21,
) -> pd.DataFrame:
    """
    Build the complete dispersion trading signal dataframe.
    
    This implements the regression-based prediction:
    Predicted_Realized_Dispersion = f(Implied_Dispersion, Idio_Vol_Signal)
    
    Returns:
        DataFrame with columns for all signal components
    """
    results = []
    
    for date in index_implied_vol_history.index[lookback:]:
        try:
            # Current implied dispersion
            idx_iv = index_implied_vol_history.loc[date]
            const_ivs = constituent_implied_vols_history.loc[date]
            
            implied_metrics = compute_implied_dispersion(idx_iv, const_ivs, weights)
            
            # Realized dispersion (backward-looking)
            past_date = date - pd.Timedelta(days=lookback)
            mask = constituent_returns_history.index <= date
            past_mask = constituent_returns_history.index >= past_date
            valid_mask = mask & past_mask
            
            if valid_mask.sum() < lookback // 2:
                continue
                
            recent_returns = constituent_returns_history.loc[valid_mask]
            recent_index = index_returns_history.loc[valid_mask]
            
            realized_metrics = compute_realized_dispersion(
                recent_returns, recent_index, weights, lookback=valid_mask.sum()
            )
            
            # Idiosyncratic volatility signal
            idio_ratio = compute_index_idiosyncratic_ratio(
                recent_returns, recent_index, weights, lookback=valid_mask.sum()
            )
            
            results.append({
                "date": date,
                "index_iv": implied_metrics["index_iv"],
                "weighted_constituent_iv": implied_metrics["weighted_constituent_iv"],
                "implied_dispersion": implied_metrics["dispersion_spread"],
                "implied_correlation": implied_metrics["implied_correlation"],
                "realized_dispersion": realized_metrics["realized_dispersion"],
                "idio_vol_ratio": idio_ratio.iloc[-1] if len(idio_ratio) > 0 else np.nan,
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results).set_index("date")


def run_dispersion_regression(
    signal_df: pd.DataFrame,
    forward_window: int = 21,
) -> Dict[str, float]:
    """
    Run the predictive regression:
    
    Realized_Dispersion_forward = gamma_0 
                                  + gamma_1 * Implied_Dispersion_current
                                  + gamma_2 * Idio_Vol_Ratio_current
                                  + eta
    
    This tells us how much predictive power our signals have.
    """
    # Create forward-looking realized dispersion
    signal_df["realized_dispersion_forward"] = signal_df["realized_dispersion"].shift(-forward_window)
    
    # Clean data
    reg_data = signal_df[["realized_dispersion_forward", "implied_dispersion", "idio_vol_ratio"]].dropna()
    
    if len(reg_data) < 30:
        raise ValueError(f"Insufficient data ({len(reg_data)} points) for regression")
    
    y = reg_data["realized_dispersion_forward"].values
    X = np.column_stack([
        np.ones(len(reg_data)),  # Intercept
        reg_data["implied_dispersion"].values,
        reg_data["idio_vol_ratio"].values,
    ])
    
    coeffs, residuals, rank, s = lstsq(X, y, rcond=None)
    
    gamma_0, gamma_1, gamma_2 = coeffs
    
    # Calculate R-squared
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        "intercept": gamma_0,
        "coef_implied_dispersion": gamma_1,
        "coef_idio_vol_ratio": gamma_2,
        "r_squared": r_squared,
        "n_observations": len(reg_data),
    }


def generate_dispersion_signals(
    signal_df: pd.DataFrame,
    params: DispersionParams,
) -> pd.Series:
    """
    Generate trading signals based on:
    1. Implied dispersion level (z-score)
    2. Idiosyncratic volatility regime
    
    Signal logic:
    - Short dispersion when implied is high AND idio vol is elevated
      (expecting correlation to increase, index vol to converge down to constituents)
    - Long dispersion when implied is low AND idio vol is low
      (expecting correlation to decrease)
    """
    signals = pd.Series(0, index=signal_df.index, name="signal")
    
    # Calculate z-scores
    implied_disp_zscore = (
        signal_df["implied_dispersion"] - signal_df["implied_dispersion"].rolling(params.lookback_window).mean()
    ) / signal_df["implied_dispersion"].rolling(params.lookback_window).std()
    
    idio_vol_percentile = signal_df["idio_vol_ratio"].rolling(params.lookback_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )
    
    for date in signal_df.index:
        zscore = implied_disp_zscore.loc[date]
        idio_pct = idio_vol_percentile.loc[date]
        
        if not np.isfinite(zscore) or not np.isfinite(idio_pct):
            continue
        
        # Short dispersion: implied dispersion is rich AND idio vol is high
        if params.trade_mode == "short_dispersion":
            if zscore > params.signal_threshold and idio_pct > params.idio_vol_percentile_threshold:
                signals.loc[date] = -1  # Short dispersion
            elif zscore < -params.signal_threshold and idio_pct < (100 - params.idio_vol_percentile_threshold):
                signals.loc[date] = 1  # Long dispersion (unwind)
        
        # Long dispersion: implied dispersion is cheap AND idio vol is low
        elif params.trade_mode == "long_dispersion":
            if zscore < -params.signal_threshold and idio_pct < (100 - params.idio_vol_percentile_threshold):
                signals.loc[date] = 1  # Long dispersion
            elif zscore > params.signal_threshold and idio_pct > params.idio_vol_percentile_threshold:
                signals.loc[date] = -1  # Short dispersion (unwind)
    
    return signals


def simulate_dispersion_pnl(
    signal_df: pd.DataFrame,
    signals: pd.Series,
    holding_period: int = 5,
    dispersion_notional: float = 1_000_000,
) -> pd.DataFrame:
    """
    Simulate P&L from dispersion trades.
    
    Position structure for short dispersion:
    - Sell index options (e.g., SPX straddle) = -1x notional * index_vega
    - Buy constituent options proportional to weights = +1x notional * weighted vega
    
    Simplified P&L approximation:
    P&L ≈ Notional * (Realized_Dispersion_Entry - Realized_Dispersion_Exit)
    """
    trades = pd.DataFrame({
        "signal": signals,
        "implied_dispersion": signal_df["implied_dispersion"],
        "realized_dispersion": signal_df["realized_dispersion"],
    }).copy()
    
    trades["entry_dispersion"] = np.nan
    trades["exit_dispersion"] = np.nan
    trades["pnl"] = 0.0
    trades["position_active"] = False
    
    position = 0
    entry_idx = None
    entry_dispersion = None
    
    for i, (date, row) in enumerate(trades.iterrows()):
        current_signal = row["signal"]
        
        # New entry
        if position == 0 and current_signal != 0:
            position = current_signal
            entry_idx = i
            entry_dispersion = row["implied_dispersion"]
            trades.at[date, "entry_dispersion"] = entry_dispersion
            trades.at[date, "position_active"] = True
        
        # Exit: either signal reversal or holding period exceeded
        elif position != 0:
            should_exit = (
                (current_signal != 0 and current_signal != position)  # Signal flip
                or (i - entry_idx >= holding_period)  # Time exit
            )
            
            if should_exit:
                # Calculate P&L
                # For short dispersion: profit when realized < implied (at entry)
                exit_dispersion = row["realized_dispersion"]
                trades.at[date, "exit_dispersion"] = exit_dispersion
                
                # P&L = position * (entry_implied - exit_realized) * notional
                # Short position (-1) profits when exit_realized < entry_implied
                pnl = -position * (exit_dispersion - entry_dispersion) * dispersion_notional
                trades.at[date, "pnl"] = pnl
                
                # Reset
                position = 0
                entry_idx = None
                entry_dispersion = None
            else:
                trades.at[date, "position_active"] = True
    
    return trades


def print_regression_summary(reg_results: RegressionResult) -> None:
    """Pretty print the return decomposition results."""
    print(f"\n=== Return Decomposition: {reg_results.stock_symbol} ===")
    print(f"  Alpha (Jensen's):    {reg_results.alpha:.4f} ({reg_results.alpha*252:.2%} annualized)")
    print(f"  Beta:                {reg_results.beta:.3f}")
    print(f"  R-squared:           {reg_results.r_squared:.3f}")
    print(f"  Total Volatility:    {reg_results.total_volatility:.2%} (annualized)")
    print(f"  Systematic Vol:      {reg_results.systematic_volatility:.2%} ({reg_results.systematic_volatility/reg_results.total_volatility:.1%} of total)")
    print(f"  Idiosyncratic Vol:   {reg_results.idio_volatility:.2%} ({reg_results.idio_volatility/reg_results.total_volatility:.1%} of total) ← KEY METRIC")


def print_dispersion_metrics(dispersion_metrics: Dict[str, float]) -> None:
    """Pretty print dispersion metrics."""
    print(f"\n=== Dispersion Metrics ===")
    print(f"  Index IV:            {dispersion_metrics['index_iv']:.2%}")
    print(f"  Weighted Stock IV:   {dispersion_metrics['weighted_constituent_iv']:.2%}")
    print(f"  Dispersion Spread:   {dispersion_metrics['dispersion_spread']:.2%} (positive = index more expensive)")
    print(f"  Implied Correlation: {dispersion_metrics['implied_correlation']:.3f}")


def print_predictive_regression(pred_reg: Dict[str, float]) -> None:
    """Print the predictive regression results."""
    print(f"\n=== Predictive Regression Results ===")
    print(f"  Realized_Dispersion_t+21 = ")
    print(f"      {pred_reg['intercept']:+.4f}")
    print(f"    + {pred_reg['coef_implied_dispersion']:+.4f} * Implied_Dispersion_t")
    print(f"    + {pred_reg['coef_idio_vol_ratio']:+.4f} * Idio_Vol_Ratio_t")
    print(f"  ")
    print(f"  R-squared:           {pred_reg['r_squared']:.3f}")
    print(f"  Observations:        {pred_reg['n_observations']}")


# =============
# MAIN WORKFLOW
# =============

def main_example():
    """
    Example workflow for idiosyncratic volatility dispersion trading.
    
    This demonstrates the full pipeline:
    1. Decompose returns for each stock
    2. Compute aggregate idiosyncratic volatility ratio
    3. Calculate implied vs realized dispersion
    4. Build predictive regression
    5. Generate trading signals
    6. Simulate P&L
    """
    print("=" * 60)
    print("IDIOSYNCRATIC VOLATILITY DISPERSION TRADING")
    print("Based on Assenagon's Approach")
    print("=" * 60)
    
    print("\n--- KEY LINEAR REGRESSION EQUATIONS ---")
    print("""
    1. Return Decomposition:
       r_i,t = α_i + β_i × r_m,t + ε_i,t
       
       Where ε_i,t = idiosyncratic return
    
    2. Idiosyncratic Volatility:
       σ_idio,i = √(Var(ε_i))
    
    3. Index Variance Decomposition:
       σ_m² = Σ_i w_i²σ_i² + Σ_i≠j w_i w_j ρ_ij σ_i σ_j
    
    4. Implied Dispersion:
       Dispersion_implied = σ_index,impl - Σ_i w_i × σ_i,impl
    
    5. Predictive Regression:
       Realized_Dispersion_t+21 = γ_0 + γ_1 × Implied_Dispersion_t + γ_2 × IV_Signal_t + η
    """)
    
    # Note: In practice, you would load actual data here
    print("\nTo use this module:")
    print("  1. Load constituent returns and index returns")
    print("  2. Load constituent implied volatilities and index IV")
    print("  3. Run decompose_returns() for each stock")
    print("  4. Run build_dispersion_signal() for time series")
    print("  5. Run run_dispersion_regression() for predictive power")
    print("  6. Run generate_dispersion_signals() for trade timing")
    print("  7. Run simulate_dispersion_pnl() for backtesting")


if __name__ == "__main__":
    main_example()
