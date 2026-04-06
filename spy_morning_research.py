"""
Required input format
---------------------
Index:
- pandas `DatetimeIndex`
- timezone-aware
- intraday timestamps in `America/New_York` (or any timezone that can be converted to it)
- each row should represent one completed bar

Columns:
- `spy_open`
- `spy_high`
- `spy_low`
- `spy_close`
- `spy_volume`
- `spx_close`

Notes:
- 1-minute or 5-minute bars are ideal.
- If you do not have SPX data, copy `spy_close` into `spx_close` and use SPY as both
  the signal source and the traded instrument.
- The signal is built from the early SPX move after the US cash open.
- The trade is executed on SPY.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


NEW_YORK_TZ = "America/New_York"


@dataclass(frozen=True)
class StrategyParams:
    signal_reference: str
    mode: str
    filter_name: str
    signal_start_time: str
    signal_end_time: str
    entry_time: str
    exit_time: str
    move_threshold: float
    cost_bps_per_side: float = 1.0


def prepare_intraday_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "spy_open",
        "spy_high",
        "spy_low",
        "spy_close",
        "spy_volume",
        "spx_close",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The dataframe index must be a pandas DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("The dataframe index must be timezone-aware.")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in the dataframe index.")

    clean = df.sort_index().copy()
    clean = clean.tz_convert(NEW_YORK_TZ)
    clean = clean.astype(
        {
            "spy_open": float,
            "spy_high": float,
            "spy_low": float,
            "spy_close": float,
            "spy_volume": float,
            "spx_close": float,
        }
    )
    clean["session_date"] = clean.index.date
    clean["spx_rsi"] = compute_rsi(clean["spx_close"], period=14)
    clean["spx_macd"], clean["spx_macd_signal"], clean["spx_macd_hist"] = compute_macd(
        clean["spx_close"],
        fast=12,
        slow=26,
        signal=9,
    )
    return clean


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)


def _session_timestamp(session_date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{session_date} {hhmm}", tz=NEW_YORK_TZ)


def _first_at_or_after(session: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    subset = session.loc[session.index >= timestamp]
    if subset.empty:
        return None
    return subset.iloc[0]


def _first_after(session: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    subset = session.loc[session.index > timestamp]
    if subset.empty:
        return None
    return subset.iloc[0]


def _last_at_or_before(session: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    subset = session.loc[session.index <= timestamp]
    if subset.empty:
        return None
    return subset.iloc[-1]


# =========================
# RESEARCH REGION
# =========================


def build_daily_dataset(
    intraday: pd.DataFrame,
    signal_start_time: str,
    signal_end_time: str,
    entry_time: str,
    exit_time: str,
    signal_reference: str,
) -> pd.DataFrame:
    if signal_reference not in {"session_open", "prev_close"}:
        raise ValueError("signal_reference must be either 'session_open' or 'prev_close'.")

    session_closes = intraday.groupby("session_date", sort=True)["spx_close"].last()
    prev_closes = session_closes.shift(1)

    rows = []
    for session_date, session in intraday.groupby("session_date", sort=True):
        signal_start_ts = _session_timestamp(session_date, signal_start_time)
        signal_end_ts = _session_timestamp(session_date, signal_end_time)
        entry_ts = _session_timestamp(session_date, entry_time)
        exit_ts = _session_timestamp(session_date, exit_time)

        signal_start_row = _first_at_or_after(session, signal_start_ts)
        signal_end_row = _last_at_or_before(session, signal_end_ts)

        if signal_start_row is None or signal_end_row is None:
            continue
        if signal_end_row.name < signal_start_row.name:
            continue

        if entry_ts <= signal_end_ts:
            entry_row = _first_after(session, signal_end_ts)
        else:
            entry_row = _first_at_or_after(session, entry_ts)
        exit_row = _first_at_or_after(session, exit_ts)

        if entry_row is None or exit_row is None:
            continue
        if exit_row.name <= entry_row.name:
            continue

        prev_close = prev_closes.get(session_date)
        if pd.isna(prev_close) and signal_reference == "prev_close":
            continue

        if signal_reference == "session_open":
            reference_price = float(signal_start_row["spx_close"])
        else:
            reference_price = float(prev_close)

        signal_end_price = float(signal_end_row["spx_close"])
        early_return = signal_end_price / reference_price - 1.0

        rows.append(
            {
                "session_date": session_date,
                "signal_reference": signal_reference,
                "signal_start_price": float(signal_start_row["spx_close"]),
                "signal_end_price": signal_end_price,
                "prev_close": np.nan if pd.isna(prev_close) else float(prev_close),
                "early_return": early_return,
                "gap_return": (
                    np.nan
                    if pd.isna(prev_close)
                    else float(signal_start_row["spx_close"]) / float(prev_close) - 1.0
                ),
                "spx_rsi": float(signal_end_row["spx_rsi"]),
                "spx_macd_hist": float(signal_end_row["spx_macd_hist"]),
                "entry_timestamp": entry_row.name,
                "exit_timestamp": exit_row.name,
                "entry_price": float(entry_row["spy_open"]),
                "exit_price": float(exit_row["spy_open"]),
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    return daily.set_index("session_date").sort_index()


def generate_signals(daily: pd.DataFrame, params: StrategyParams) -> pd.Series:
    signals = []

    for _, row in daily.iterrows():
        early_return = row["early_return"]
        abs_move = abs(early_return)

        if not np.isfinite(abs_move) or abs_move < params.move_threshold:
            signals.append(0)
            continue

        if params.mode == "momentum":
            direction = 1 if early_return > 0 else -1
        elif params.mode == "mean_reversion":
            direction = -1 if early_return > 0 else 1
        else:
            raise ValueError("mode must be either 'momentum' or 'mean_reversion'.")

        if _passes_filter(row=row, direction=direction, params=params):
            signals.append(direction)
        else:
            signals.append(0)

    return pd.Series(signals, index=daily.index, name="signal")


def _passes_filter(row: pd.Series, direction: int, params: StrategyParams) -> bool:
    if params.filter_name == "none":
        return True

    if params.filter_name == "rsi":
        if params.mode == "momentum":
            return (direction == 1 and row["spx_rsi"] >= 55) or (
                direction == -1 and row["spx_rsi"] <= 45
            )
        return (direction == 1 and row["spx_rsi"] <= 30) or (
            direction == -1 and row["spx_rsi"] >= 70
        )

    if params.filter_name == "macd":
        if params.mode == "momentum":
            return (direction == 1 and row["spx_macd_hist"] >= 0) or (
                direction == -1 and row["spx_macd_hist"] <= 0
            )
        return (direction == 1 and row["spx_macd_hist"] <= 0) or (
            direction == -1 and row["spx_macd_hist"] >= 0
        )

    raise ValueError("filter_name must be one of: 'none', 'rsi', 'macd'.")


def research_parameter_grid(
    intraday: pd.DataFrame,
    train_fraction: float = 0.70,
    min_train_trades: int = 20,
) -> pd.DataFrame:
    signal_references = ["session_open", "prev_close"]
    modes = ["momentum", "mean_reversion"]
    filters = ["none", "rsi", "macd"]
    signal_end_times = ["09:45", "10:00", "10:15"]
    exit_times = ["11:00", "11:30", "12:00"]
    move_thresholds = [0.0010, 0.0015, 0.0020, 0.0030]

    results = []
    for signal_reference in signal_references:
        for signal_end_time in signal_end_times:
            daily = build_daily_dataset(
                intraday=intraday,
                signal_start_time="09:30",
                signal_end_time=signal_end_time,
                entry_time=signal_end_time,
                exit_time="12:00",
                signal_reference=signal_reference,
            )
            if daily.empty:
                continue

            train_daily, _ = split_train_test(daily, train_fraction=train_fraction)
            if len(train_daily) < 30:
                continue

            for exit_time in exit_times:
                daily_for_exit = build_daily_dataset(
                    intraday=intraday,
                    signal_start_time="09:30",
                    signal_end_time=signal_end_time,
                    entry_time=signal_end_time,
                    exit_time=exit_time,
                    signal_reference=signal_reference,
                )
                if daily_for_exit.empty:
                    continue

                train_daily, test_daily = split_train_test(
                    daily_for_exit,
                    train_fraction=train_fraction,
                )
                if train_daily.empty or test_daily.empty:
                    continue

                for mode in modes:
                    for filter_name in filters:
                        for move_threshold in move_thresholds:
                            params = StrategyParams(
                                signal_reference=signal_reference,
                                mode=mode,
                                filter_name=filter_name,
                                signal_start_time="09:30",
                                signal_end_time=signal_end_time,
                                entry_time=signal_end_time,
                                exit_time=exit_time,
                                move_threshold=move_threshold,
                                cost_bps_per_side=1.0,
                            )

                            _, train_metrics = run_backtest(train_daily, params)
                            if train_metrics["n_trades"] < min_train_trades:
                                continue

                            _, test_metrics = run_backtest(test_daily, params)
                            full_trades, full_metrics = run_backtest(daily_for_exit, params)

                            results.append(
                                {
                                    **asdict(params),
                                    "n_days_full": int(len(daily_for_exit)),
                                    "n_trades_full": int(full_metrics["n_trades"]),
                                    "train_total_return": train_metrics["total_return"],
                                    "train_sharpe": train_metrics["sharpe"],
                                    "train_win_rate": train_metrics["win_rate"],
                                    "test_total_return": test_metrics["total_return"],
                                    "test_sharpe": test_metrics["sharpe"],
                                    "test_win_rate": test_metrics["win_rate"],
                                    "full_total_return": full_metrics["total_return"],
                                    "full_sharpe": full_metrics["sharpe"],
                                    "full_max_drawdown": full_metrics["max_drawdown"],
                                    "full_profit_factor": full_metrics["profit_factor"],
                                    "full_avg_trade": full_metrics["avg_trade"],
                                    "last_trade_date": (
                                        np.nan if full_trades.empty else full_trades.index.max()
                                    ),
                                }
                            )

    if not results:
        raise ValueError("No valid strategy configuration was found on the provided data.")

    ranked = pd.DataFrame(results).sort_values(
        by=["train_sharpe", "test_sharpe", "full_total_return"],
        ascending=False,
    )
    return ranked.reset_index(drop=True)


def split_train_test(
    daily: pd.DataFrame,
    train_fraction: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty:
        return daily.copy(), daily.copy()

    unique_dates = pd.Index(daily.index).sort_values()
    split_idx = max(1, int(len(unique_dates) * train_fraction))
    split_idx = min(split_idx, len(unique_dates) - 1)

    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    train_daily = daily.loc[train_dates]
    test_daily = daily.loc[test_dates]
    return train_daily, test_daily


# =========================
# BACKTEST REGION
# =========================


def run_backtest(
    daily: pd.DataFrame,
    params: StrategyParams,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if daily.empty:
        return daily.copy(), empty_metrics()

    trades = daily.copy()
    trades["signal"] = generate_signals(trades, params)
    trades["gross_return"] = np.where(
        trades["signal"] != 0,
        trades["signal"] * (trades["exit_price"] / trades["entry_price"] - 1.0),
        0.0,
    )
    round_trip_cost = 2.0 * params.cost_bps_per_side / 10000.0
    trades["net_return"] = np.where(
        trades["signal"] != 0,
        trades["gross_return"] - round_trip_cost,
        0.0,
    )
    trades["equity_curve"] = (1.0 + trades["net_return"]).cumprod()
    trades["drawdown"] = trades["equity_curve"] / trades["equity_curve"].cummax() - 1.0

    metrics = evaluate_trades(trades)
    return trades, metrics


def evaluate_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return empty_metrics()

    all_returns = trades["net_return"].fillna(0.0)
    active_returns = trades.loc[trades["signal"] != 0, "net_return"].copy()

    if active_returns.empty:
        return {
            "n_days": float(len(trades)),
            "n_trades": 0.0,
            "hit_ratio": 0.0,
            "total_return": 0.0,
            "avg_trade": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    total_return = float((1.0 + all_returns).prod() - 1.0)
    avg_trade = float(active_returns.mean())
    win_rate = float((active_returns > 0).mean())

    std = float(all_returns.std(ddof=0))
    sharpe = 0.0 if std == 0.0 else float(np.sqrt(252.0) * all_returns.mean() / std)

    gross_profit = float(active_returns[active_returns > 0].sum())
    gross_loss = float(active_returns[active_returns < 0].sum())
    if gross_loss == 0.0:
        profit_factor = np.inf
    else:
        profit_factor = gross_profit / abs(gross_loss)

    max_drawdown = float(trades["drawdown"].min())

    return {
        "n_days": float(len(trades)),
        "n_trades": float(len(active_returns)),
        "hit_ratio": float(len(active_returns) / len(trades)),
        "total_return": total_return,
        "avg_trade": avg_trade,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": float(profit_factor),
    }


def empty_metrics() -> Dict[str, float]:
    return {
        "n_days": 0.0,
        "n_trades": 0.0,
        "hit_ratio": 0.0,
        "total_return": 0.0,
        "avg_trade": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
    }


def print_research_summary(research_results: pd.DataFrame, top_n: int = 10) -> None:
    display_cols = [
        "mode",
        "filter_name",
        "signal_reference",
        "signal_end_time",
        "exit_time",
        "move_threshold",
        "train_sharpe",
        "test_sharpe",
        "full_total_return",
        "full_max_drawdown",
        "n_trades_full",
    ]

    print("\n=== TOP RESEARCH RESULTS ===")
    print(research_results.loc[:, display_cols].head(top_n).to_string(index=False))


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    formatted = {
        "n_days": int(metrics["n_days"]),
        "n_trades": int(metrics["n_trades"]),
        "hit_ratio": f"{metrics['hit_ratio']:.2%}",
        "total_return": f"{metrics['total_return']:.2%}",
        "avg_trade": f"{metrics['avg_trade']:.4%}",
        "win_rate": f"{metrics['win_rate']:.2%}",
        "sharpe": f"{metrics['sharpe']:.2f}",
        "max_drawdown": f"{metrics['max_drawdown']:.2%}",
        "profit_factor": f"{metrics['profit_factor']:.2f}",
    }

    print(f"\n=== {title} ===")
    for key, value in formatted.items():
        print(f"{key:>15}: {value}")


def main(df: pd.DataFrame) -> Dict[str, object]:
    intraday = prepare_intraday_data(df)
    research_results = research_parameter_grid(intraday)
    best_row = research_results.iloc[0]

    best_params = StrategyParams(
        signal_reference=best_row["signal_reference"],
        mode=best_row["mode"],
        filter_name=best_row["filter_name"],
        signal_start_time=best_row["signal_start_time"],
        signal_end_time=best_row["signal_end_time"],
        entry_time=best_row["entry_time"],
        exit_time=best_row["exit_time"],
        move_threshold=float(best_row["move_threshold"]),
        cost_bps_per_side=float(best_row["cost_bps_per_side"]),
    )

    daily = build_daily_dataset(
        intraday=intraday,
        signal_start_time=best_params.signal_start_time,
        signal_end_time=best_params.signal_end_time,
        entry_time=best_params.entry_time,
        exit_time=best_params.exit_time,
        signal_reference=best_params.signal_reference,
    )
    train_daily, test_daily = split_train_test(daily, train_fraction=0.70)

    train_trades, train_metrics = run_backtest(train_daily, best_params)
    test_trades, test_metrics = run_backtest(test_daily, best_params)
    full_trades, full_metrics = run_backtest(daily, best_params)

    print_research_summary(research_results, top_n=10)
    print("\n=== CHOSEN STRATEGY ===")
    print(pd.Series(asdict(best_params)).to_string())
    print_metrics("TRAIN BACKTEST", train_metrics)
    print_metrics("TEST BACKTEST", test_metrics)
    print_metrics("FULL SAMPLE BACKTEST", full_metrics)

    return {
        "best_params": best_params,
        "research_results": research_results,
        "train_trades": train_trades,
        "test_trades": test_trades,
        "full_trades": full_trades,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
    }


if __name__ == "__main__":
    print("Load your intraday dataframe and call: results = main(df)")
