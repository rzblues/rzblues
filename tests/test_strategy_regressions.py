import unittest
from unittest.mock import Mock, patch

import pandas as pd

from backtest import _CASH_RATE_WEEKLY, simulate
from panic4factor import alerts, fetch
from panic4factor.state import TradingState


def _weekly_frame(rows):
    df = pd.DataFrame(rows, index=pd.to_datetime([r.pop("date") for r in rows]))
    df.attrs["idx_col"] = "qqq"
    return df


def _base_row(date, price):
    return {
        "date": date,
        "qqq": price,
        "vix": 10.0,
        "fg_approx": 100.0,
        "aaii_spread": 0.0,
        "naaim": 100.0,
        "ma200": 100.0,
        "high52w": max(price, 100.0),
        "hyg": 100.0,
        "lqd": 100.0,
        "hyg_ma200": 100.0,
        "hyg_4w": 100.0,
        "lqd_4w": 100.0,
        "tlt_vol10": 0.0,
    }


def _panic_row(date, price):
    row = _base_row(date, price)
    row.update({
        "vix": 40.0,
        "fg_approx": 10.0,
        "aaii_spread": -50.0,
        "naaim": 10.0,
        "high52w": 150.0,
    })
    return row


class BacktestRegressionTest(unittest.TestCase):
    def test_entry_signal_does_not_earn_prior_week_return(self):
        df = _weekly_frame([
            _base_row("2020-01-03", 100.0),
            _panic_row("2020-01-10", 110.0),
        ])

        trades, equity = simulate(df, cost_bps=0.0)

        self.assertEqual(trades[0].entry_date.isoformat(), "2020-01-10")
        self.assertEqual(trades[0].entry_price, 110.0)
        self.assertAlmostEqual(float(equity.iloc[-1]), 1 + _CASH_RATE_WEEKLY, places=9)

    def test_exit_signal_keeps_return_until_exit_close(self):
        df = _weekly_frame([
            _panic_row("2020-01-03", 100.0),
            _base_row("2020-01-10", 110.0),
        ])

        trades, equity = simulate(df, cost_bps=0.0)

        self.assertEqual(trades[0].exit_date.isoformat(), "2020-01-10")
        self.assertAlmostEqual(float(equity.iloc[-1]), 1.10, places=9)


class SchedulerRegressionTest(unittest.TestCase):
    def test_exit_alerts_can_format_after_state_closes(self):
        state = TradingState(
            allocation=1.0,
            peak_allocation=1.0,
            avg_entry_price=100.0,
            entry_date="2020-01-01",
            last_tier="TIER_4_MAX",
        )
        entry_price_snapshot = state.avg_entry_price

        fired = state.check_exits(price=130.0, vix=15.0, fg=60.0, high_52w=132.0)
        self.assertIsNone(state.avg_entry_price)
        self.assertEqual([n for n, _ in fired], [1, 2, 3])

        snap = type("Snap", (), {"ticker": "QQQ", "index_price": 130.0})()
        messages = [
            alerts.fmt_exit(n, reason, snap, entry_price_snapshot)
            for n, reason in fired
        ]
        self.assertIn("Tranche 3/3", messages[-1])


class FetchRegressionTest(unittest.TestCase):
    def test_naaim_dynamic_excel_discovery(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.text = (
            '<a href="https://naaim.org/wp-content/uploads/2026/05/'
            'USE_Data-since-Inception_2026-05-06.xlsx">download</a>'
        )

        with patch("panic4factor.fetch.requests.get", return_value=response):
            self.assertEqual(
                fetch._find_naaim_excel_url(),
                "https://naaim.org/wp-content/uploads/2026/05/"
                "USE_Data-since-Inception_2026-05-06.xlsx",
            )


if __name__ == "__main__":
    unittest.main()
