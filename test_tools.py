"""Offline tests for tools.py (no network, sleeps mocked):   python -m unittest test_tools"""
import json
import os
import unittest
from unittest import mock

import httpx

import tools
from tools import RetryableError, with_retry

ATOM_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T00:00:00Z</published>
<title>A  World
  Model</title><summary> Learns   dynamics.
</summary></entry></feed>"""


def response(status=200, *, json_body=None, text=None, headers=None, content_type="application/json"):
    headers = {"content-type": content_type, **(headers or {})}
    request = httpx.Request("GET", "https://example.org")
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers, request=request)
    return httpx.Response(status, text=text or "", headers=headers, request=request)


def sse(message):
    return response(text=f"event: message\ndata: {json.dumps(message)}\n\n", content_type="text/event-stream")


class Flaky:
    """Raises the given errors in order, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@mock.patch("tools.time.sleep")
class WithRetryTest(unittest.TestCase):
    def test_success_without_sleep(self, sleep):
        self.assertEqual(with_retry(Flaky()), "ok")
        sleep.assert_not_called()

    def test_retries_then_succeeds_with_capped_jittered_backoff(self, sleep):
        fn = Flaky(*[RetryableError("x")] * 4)
        self.assertEqual(with_retry(fn, attempts=5, base=1.0, cap=5.0), "ok")
        delays = [c.args[0] for c in sleep.call_args_list]
        self.assertEqual(fn.calls, 5)
        for attempt, delay in enumerate(delays):  # equal jitter: between half and all of min(base*2**n, cap)
            ceiling = min(2 ** attempt, 5.0)
            self.assertTrue(ceiling / 2 <= delay <= ceiling, (attempt, delay))

    def test_jitter_is_random(self, sleep):
        for _ in range(20):
            with_retry(Flaky(RetryableError("x")), base=8.0, cap=60.0)
        self.assertGreater(len({c.args[0] for c in sleep.call_args_list}), 1)

    def test_retry_after_is_used_and_capped(self, sleep):
        with_retry(Flaky(RetryableError("x", retry_after=7), RetryableError("x", retry_after=500)), cap=30.0)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [7, 30.0])

    def test_gives_up_without_sleeping_after_last_attempt(self, sleep):
        fn = Flaky(*[RetryableError("boom")] * 3)
        with self.assertRaises(RetryableError):
            with_retry(fn, attempts=3)
        self.assertEqual((fn.calls, sleep.call_count), (3, 2))

    def test_other_errors_are_not_retried(self, sleep):
        fn = Flaky(ValueError("bug"))
        with self.assertRaises(ValueError):
            with_retry(fn)
        self.assertEqual((fn.calls, sleep.call_count), (1, 0))


@mock.patch("tools.time.sleep")
class RequestTest(unittest.TestCase):
    def test_retryable_status_and_retry_after(self, _):
        with mock.patch("tools.httpx.request", return_value=response(503, text="", headers={"retry-after": "4"})):
            with self.assertRaises(RetryableError) as ctx:
                tools._request("GET", "https://example.org")
        self.assertEqual(ctx.exception.retry_after, 4.0)

    def test_huge_retry_after_gives_up(self, _):
        with mock.patch("tools.httpx.request", return_value=response(429, text="", headers={"retry-after": "55022"})):
            with self.assertRaises(RuntimeError):
                tools._request("GET", "https://example.org")

    def test_transport_error_is_retryable(self, _):
        with mock.patch("tools.httpx.request", side_effect=httpx.ConnectTimeout("t")):
            with self.assertRaises(RetryableError):
                tools._request("GET", "https://example.org")

    def test_client_error_is_not_retryable(self, _):
        with mock.patch("tools.httpx.request", return_value=response(400, text="bad")):
            with self.assertRaises(httpx.HTTPStatusError):
                tools._request("GET", "https://example.org")


@mock.patch("tools.time.sleep")
class ArxivTest(unittest.TestCase):
    def test_parses_and_normalises(self, _):
        with mock.patch("tools.httpx.request", return_value=response(text=ATOM_FEED)) as request:
            records = json.loads(tools.arxiv_search.invoke({"query": "world: 'model' AND", "max_results": 99}))
        params = request.call_args.kwargs["params"]
        self.assertEqual(params["search_query"], "all:world AND all:model")
        self.assertEqual(params["max_results"], 30)
        self.assertTrue(request.call_args.args[1].startswith("https://"))
        self.assertEqual(records, [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001",
                                    "published": "2025-01-02", "title": "A World Model", "summary": "Learns dynamics."}])

    def test_empty_query_does_not_call_the_network(self, _):
        with mock.patch("tools.httpx.request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": " :'\" AND "}), "NO RESULTS")
        request.assert_not_called()

    def test_no_entries(self, _):
        feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        with mock.patch("tools.httpx.request", return_value=response(text=feed)):
            self.assertEqual(tools.arxiv_search.invoke({"query": "x"}), "NO RESULTS")

    def test_spacing_between_calls(self, sleep):
        tools._arxiv_last_call = tools.time.monotonic()
        with mock.patch("tools.httpx.request", return_value=response(text=ATOM_FEED)):
            tools.arxiv_search.invoke({"query": "x"})
        self.assertGreater(sleep.call_args_list[0].args[0], 2.5)

    def test_persistent_failure_is_an_error_string(self, _):
        with mock.patch("tools.httpx.request", return_value=response(429, text="")):
            self.assertTrue(tools.arxiv_search.invoke({"query": "x"}).startswith("ERROR: RetryableError: HTTP 429"))


class HuggingFaceTest(unittest.TestCase):
    ITEMS = [{"paper": {"id": "1", "title": "World model A", "summary": "s", "ai_summary": "short", "upvotes": 3,
                        "githubRepo": "g", "githubStars": 9, "publishedAt": "2025-01-01T00:00:00Z"}},
             {"paper": {"id": "2", "title": "Robots", "summary": "about a world model", "upvotes": 10}},
             {"paper": {"title": "no id"}}]

    def test_daily_sorted_and_filtered(self):
        with mock.patch("tools.httpx.request", return_value=response(json_body=self.ITEMS)):
            records = json.loads(tools.hf_daily_papers.invoke({"keyword": "World Model"}))
        self.assertEqual([r["id"] for r in records], ["2", "1"])
        self.assertEqual(records[1]["url"], "https://huggingface.co/papers/1")
        self.assertEqual(records[1]["summary"], "s")
        self.assertEqual(set(records[1]), {"id", "url", "published", "title", "summary", "upvotes", "github", "stars"})

    def test_search_prefers_ai_summary(self):
        with mock.patch("tools.httpx.request", return_value=response(json_body=self.ITEMS)):
            records = json.loads(tools.hf_search_papers.invoke({"query": "world model"}))
        self.assertEqual(records[0]["summary"], "short")

    def test_empty_and_error(self):
        with mock.patch("tools.httpx.request", return_value=response(json_body=[])):
            self.assertEqual(tools.hf_daily_papers.invoke({}), "NO RESULTS")
        with mock.patch("tools.httpx.request", return_value=response(400, text="bad date")):
            self.assertTrue(tools.hf_daily_papers.invoke({"date": "2026-13-45"}).startswith("ERROR: "))


@mock.patch("tools.time.sleep")
class ExaTest(unittest.TestCase):
    OK = {"jsonrpc": "2.0", "id": 1, "result": {"_meta": {"ai.exa/usage": {"costDollars": 0.007}},
                                                "content": [{"type": "text", "text": "Title: T\nURL: https://x.org"}]}}
    LIMITED = {"jsonrpc": "2.0", "id": 1, "result": {"_meta": {"ai.exa/rateLimited": True},
                                                     "content": [{"type": "text", "text": "slow down"}]}}

    def test_search_reads_sse_and_builds_objective(self, _):
        with mock.patch("tools.httpx.request", return_value=sse(self.OK)) as request:
            self.assertEqual(tools.web_search.invoke({"query": "world models"}), "Title: T\nURL: https://x.org")
        arguments = request.call_args.kwargs["json"]["params"]["arguments"]
        self.assertTrue(arguments["objective"])

    def test_rate_limit_flag_with_http_200_is_retried(self, sleep):
        with mock.patch("tools.httpx.request", side_effect=[sse(self.LIMITED), sse(self.OK)]):
            self.assertEqual(tools.web_search.invoke({"query": "q"}), "Title: T\nURL: https://x.org")
        self.assertEqual(sleep.call_count, 1)

    def test_jsonrpc_error(self, _):
        with mock.patch("tools.httpx.request", return_value=sse({"error": {"code": -1, "message": "bad args"}})):
            self.assertTrue(tools.web_fetch.invoke({"url": "https://x.org"}).startswith("ERROR: RuntimeError: Exa"))

    def test_key_is_sent_in_header_and_never_returned(self, _):
        secret = "secret-key-123456789"
        failure = httpx.ConnectError(f"failed for {tools.EXA_URL}?exaApiKey={secret}")
        with mock.patch.dict(os.environ, {"EXA_API_KEY": secret}), \
                mock.patch("tools.httpx.request", side_effect=failure) as request:
            out = tools.web_fetch.invoke({"url": "https://x.org"})
            self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], f"Bearer {secret}")
            self.assertNotIn(secret, tools._error(failure))
        self.assertTrue(out.startswith("ERROR: "))
        self.assertNotIn(secret, out)

    def test_fetch_truncates_and_validates_url(self, _):
        long_page = {"result": {"content": [{"type": "text", "text": "x" * 20_000}]}}
        with mock.patch("tools.httpx.request", return_value=sse(long_page)):
            self.assertLess(len(tools.web_fetch.invoke({"url": "https://x.org"})), 12_100)
        self.assertTrue(tools.web_fetch.invoke({"url": "file:///etc/passwd"}).startswith("ERROR: "))


class ProvenanceTest(unittest.TestCase):
    def test_canonical_url(self):
        c = tools.canonical_url
        self.assertEqual(c("https://www.Arxiv.org/abs/2501.00001v3/"), c("http://arxiv.org/abs/2501.00001"))
        self.assertEqual(c("https://en.wikipedia.org/wiki/A_(b))."), "en.wikipedia.org/wiki/A_(b)")
        self.assertNotEqual(c("https://huggingface.co/papers/2501.00001"), c("https://arxiv.org/abs/2501.00001"))

    @mock.patch("tools.time.sleep")
    def test_tools_record_returned_urls_only(self, _):
        tools.SEEN_URLS.clear()
        with mock.patch("tools.httpx.request", return_value=response(text=ATOM_FEED)):
            tools.arxiv_search.invoke({"query": "x"})
        with mock.patch("tools.httpx.request", return_value=sse(ExaTest.OK)):
            tools.web_search.invoke({"query": "q"})
        with mock.patch("tools.httpx.request", return_value=sse({"error": {"message": "bad"}})):
            tools.web_fetch.invoke({"url": "https://made-up.org/x"})
        self.assertIn(tools.canonical_url("https://arxiv.org/abs/2501.00001"), tools.SEEN_URLS)
        self.assertIn(tools.canonical_url("https://x.org"), tools.SEEN_URLS)
        self.assertNotIn(tools.canonical_url("https://made-up.org/x"), tools.SEEN_URLS)


if __name__ == "__main__":
    unittest.main()
