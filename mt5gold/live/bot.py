from __future__ import annotations
from mt5gold.core.features import feature_row, WARMUP_BARS
from mt5gold.core.types import StrategyState, Signal, Position
from mt5gold.live.safety import spread_ok, daily_halt, should_force_flat
from mt5gold.live.journal import record_decision


class LiveBot:
    def __init__(self, broker, strategy, spec, cost_cfg, exec_cfg, safety_cfg, journal_path):
        self.broker, self.strategy, self.spec = broker, strategy, spec
        self.cost_cfg, self.exec_cfg, self.safety_cfg = cost_cfg, exec_cfg, safety_cfg
        self.journal_path = journal_path

    def _open_position(self):
        ps = self.broker.positions_get(symbol=self.spec.get("symbol", "XAUUSD"))
        if not ps:
            return None
        p = ps[0]
        return Position(p["type"], p["entry_price"], p["sl"], p["tp"], p["lot"], p["entry_time"])

    def build_state(self, bar_time, bar_index, buffer_len) -> StrategyState:
        acc = self.broker.account_info()
        return StrategyState(self._open_position(), 0, buffer_len >= WARMUP_BARS,
                             bar_index, bar_time, acc["balance"], acc["equity"], 0, 0.0)

    def on_new_closed_bar(self, buffer_df) -> Signal:
        last = buffer_df.iloc[-1]
        state = self.build_state(last["time"], len(buffer_df), len(buffer_df))
        if not state.warmup_ready or not spread_ok(last["spread"], self.safety_cfg) \
           or daily_halt(state, self.safety_cfg) or should_force_flat(last["time"], self.safety_cfg):
            sig = Signal("FLAT", 0.0, None, None, {"gate": "safety_or_warmup"})
        else:
            frow = feature_row(buffer_df)
            frow["close"] = float(last["close"])   # strategy reads close; mirror backtest's feats["close"]
            sig = self.strategy.generate_signal(frow, state)
        record_decision(self.journal_path, {"bar_time": last["time"].isoformat(),
            "side": sig.side, "confidence": sig.confidence, "action": "signal"})
        return sig
