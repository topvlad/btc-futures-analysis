import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import analyze


def _sample_klines_payload():
    return [
        [
            0,
            "1",
            "2",
            "0.5",
            "1.5",
            "10",
            1,
            "10",
            1,
            "5",
            "5",
            "0",
        ],
        [
            1,
            "1.1",
            "2.1",
            "0.6",
            "1.6",
            "11",
            2,
            "11",
            1,
            "5.5",
            "5.5",
            "0",
        ],
    ]


def test_fetch_klines_forced_continuous_uses_symbol_params(monkeypatch):
    calls = []

    def fake_get(route, params):
        calls.append((route, dict(params)))
        assert route == "/fapi/v1/continuousKlines"
        assert params["pair"] == "BTCUSDT"
        assert params["contractType"] == "PERPETUAL"
        return _sample_klines_payload()

    monkeypatch.setattr(analyze, "USE_CONTINUOUS", True)
    monkeypatch.setattr(analyze, "_get", fake_get)

    df, route = analyze.fetch_klines("btcusdt", "1m", limit=2)

    assert route == "continuousKlines (forced)"
    assert not df.empty
    assert calls == [
        ("/fapi/v1/continuousKlines", {
            "interval": "1m",
            "limit": 2,
            "pair": "BTCUSDT",
            "contractType": "PERPETUAL",
        })
    ]


def test_fetch_klines_fallback_uses_symbol_specific_pair(monkeypatch):
    symbols = (
        ("BTCUSDT", "BTCUSDT"),
        ("BTCUSDC", "BTCUSDC"),
    )

    for symbol, expected_pair in symbols:
        calls = []

        def fake_get(route, params):
            params_copy = dict(params)
            calls.append((route, params_copy))
            if route == "/fapi/v1/klines":
                raise RuntimeError("primary endpoint unavailable")
            assert route == "/fapi/v1/continuousKlines"
            assert params_copy["pair"] == expected_pair
            assert params_copy["contractType"] == "PERPETUAL"
            return _sample_klines_payload()

        monkeypatch.setattr(analyze, "USE_CONTINUOUS", False)
        monkeypatch.setattr(analyze, "_get", fake_get)

        df, route = analyze.fetch_klines(symbol, "5m", limit=2)

        assert route == "continuousKlines (fallback)"
        assert not df.empty
        assert calls[0][0] == "/fapi/v1/klines"
        assert calls[1][0] == "/fapi/v1/continuousKlines"
        assert calls[1][1]["pair"] == expected_pair

