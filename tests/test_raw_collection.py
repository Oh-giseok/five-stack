"""Verify credentials, raw preservation and bounded requests without live APIs."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config
from oinq.collectors.raw import AccessError, RawClient, comtrade_request, customs_request, month_ranges, months_in_range


def response(content, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = content
    result.headers["Content-Type"] = "application/json"
    return result


class RawCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(ROOT)

    def client(self, folder):
        config = deepcopy(self.config)
        config["sources"]["request_defaults"].update(minimum_interval_seconds=0, max_retries=0)
        return RawClient(Path(folder), config, "test_snapshot")

    def test_windows_cover_exactly_the_requested_165_months(self):
        windows = month_ranges(self.config)
        months = [month for start, end in windows for month in months_in_range(start, end)]
        self.assertEqual(len(windows), 14)
        self.assertEqual(len(months), 165)
        self.assertEqual(len(set(months)), 165)
        self.assertEqual((months[0], months[-1]), ("201301", "202609"))
        self.assertTrue(all(len(months_in_range(start, end)) <= 12 for start, end in windows))

    def test_raw_bytes_retained_and_resume_does_not_request_again(self):
        content = b'{"data": [{"value": null}], "spacing":  1}'
        with tempfile.TemporaryDirectory() as folder:
            client = self.client(folder)
            session = Mock()
            session.get.return_value = response(content)
            with patch.object(client, "session", return_value=session):
                actual, first = client.fetch("world_bank", "raw", "https://api.worldbank.org/v2", {"page": 1})
                cached, second = client.fetch("world_bank", "raw", "https://api.worldbank.org/v2", {"page": 1})
            self.assertEqual(actual, content)
            self.assertEqual(cached, content)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertFalse(first["preprocessing_applied"])
            session.get.assert_called_once()

    def test_echoed_key_removed_and_never_saved_in_request_metadata(self):
        secret = "dummy-test-key"
        with tempfile.TemporaryDirectory() as folder:
            client = self.client(folder)
            session = Mock()
            session.get.return_value = response(("error=" + secret).encode())
            with patch.object(client, "session", return_value=session):
                body, meta = client.fetch("comtrade", "echo", "https://comtradeapi.un.org/data/v1/get/C/M/HS",
                                           {"period": "202501"}, headers={"Ocp-Apim-Subscription-Key": secret})
            self.assertNotIn(secret, body.decode())
            self.assertNotIn(secret, json.dumps(meta))
            self.assertTrue(meta["secret_echo_redacted"])

    def test_changed_scope_cannot_silently_reuse_or_overwrite_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.client(folder)
            session = Mock()
            session.get.return_value = response(b'{"data": []}')
            with patch.object(client, "session", return_value=session):
                client.fetch("world_bank", "same_name", "https://api.worldbank.org/v2", {"date": "2024"})
                with self.assertRaises(AccessError):
                    client.fetch("world_bank", "same_name", "https://api.worldbank.org/v2", {"date": "2025"})
            session.get.assert_called_once()

    def test_unauthorized_request_stops_without_fallback_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.client(folder)
            session = Mock()
            session.get.return_value = response(b'{"error":"unauthorized"}', 401)
            with patch.object(client, "session", return_value=session):
                with self.assertRaises(AccessError):
                    client.fetch("comtrade", "denied", "https://comtradeapi.un.org/data/v1/get/C/M/HS", {})
            session.get.assert_called_once()
            meta = json.loads((client.folder / "comtrade/denied.meta.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["transport_ok"])

    def test_network_exception_does_not_expose_secret_url(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.client(folder)
            session = Mock()
            session.get.side_effect = requests.ConnectionError("https://host/?serviceKey=private-test-token")
            with patch.object(client, "session", return_value=session):
                with self.assertRaises(AccessError) as caught:
                    client.fetch("world_bank", "failed", "https://api.worldbank.org/v2", {})
            self.assertNotIn("private-test-token", str(caught.exception))
            self.assertNotIn("private-test-token", (client.folder / "world_bank/failed.meta.json").read_text())

    def test_explicit_short_rate_limit_retries_with_same_request(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.client(folder)
            client.config["sources"]["request_defaults"]["max_retries"] = 1
            session = Mock()
            session.get.side_effect = [response(b'{"message":"Rate limit is exceeded. Try again in 1 seconds."}', 429),
                                       response(b'{"data": []}')]
            with patch.object(client, "session", return_value=session), patch("oinq.collectors.raw.time.sleep"):
                body, meta = client.fetch("world_bank", "retry", "https://api.worldbank.org/v2", {"date": "2024"})
            self.assertEqual(body, b'{"data": []}')
            self.assertEqual(meta["attempt"], 2)
            self.assertEqual(session.get.call_count, 2)
            self.assertEqual(session.get.call_args_list[0], session.get.call_args_list[1])

    def test_customs_keeps_hs10_and_total_unaggregated(self):
        document = b'<response><header><resultCode>00</resultCode></header><body><items><item><hsCd>3304991000</hsCd><expDlr>10</expDlr></item></items></body></response>'
        client = Mock()
        client.config = self.config
        client.key.return_value = "dummy"
        client.fetch.return_value = (document, {})
        rows, _ = customs_request(client, {"customs": "US"}, {"hs_code": "330499"}, "2025-01", "2025-12", "year")
        self.assertEqual(rows, [{"hsCd": "3304991000", "expDlr": "10"}])
        self.assertEqual(client.fetch.call_args.args[3]["hsSgn"], "330499")

    def test_customs_api_error_is_not_treated_as_zero_trade(self):
        client = Mock()
        client.config = self.config
        client.key.return_value = "dummy"
        client.fetch.return_value = (b'<response><header><resultCode>30</resultCode></header></response>', {})
        with self.assertRaises(AccessError):
            customs_request(client, {"customs": "US"}, {"hs_code": "330499"}, "2025-01", "2025-01", "bad")
        client.mark_api.assert_called_with("customs", "bad", {}, False)

    def test_comtrade_uses_authenticated_endpoint_and_keeps_null_weight(self):
        client = Mock()
        client.config = self.config
        client.key.return_value = "dummy"
        body = {"count": 1, "error": "", "data": [{"netWgt": None, "primaryValue": 12}]}
        client.fetch.return_value = (json.dumps(body).encode(), {})
        rows, _ = comtrade_request(client, 842, [0, 410], ["330499"], ["202501"], "M", "batch")
        self.assertIsNone(rows[0]["netWgt"])
        self.assertEqual(client.fetch.call_args.args[2], self.config["sources"]["comtrade"]["data_endpoint"])
        self.assertNotIn("preview", client.fetch.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
