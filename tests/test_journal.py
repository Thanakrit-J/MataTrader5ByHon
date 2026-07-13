from mt5gold.live.journal import record_decision, read_journal


def test_journal_append_and_read(tmp_path):
    p = tmp_path / "j.jsonl"
    record_decision(p, {"bar_time": "2020-01-01T00:00:00+00:00", "side": "BUY", "action": "placed", "retcode": 10009})
    record_decision(p, {"bar_time": "2020-01-01T00:05:00+00:00", "side": "FLAT", "action": "skip"})
    rows = read_journal(p)
    assert len(rows) == 2 and rows[0]["side"] == "BUY" and rows[1]["action"] == "skip"
