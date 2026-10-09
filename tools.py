"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree
from email.utils import parsedate_to_datetime

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 900       # a server asking for a longer pause (e.g. a daily quota) is not worth waiting for
SUMMARY_CHARS = 600
PAGE_CHARS = 12_000
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_SPACING = 3.0         # arXiv API etiquette: at least 3 s between two calls
ARXIV_OPERATORS = {"and", "or", "not", "andnot"}


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again (at most `attempts` calls in total).

    The wait is the server's Retry-After when given, else exponential backoff base * 2**attempt with jitter;
    both are capped at `cap` seconds. The last failure is re-raised without sleeping; other exceptions are not retried.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(max(exc.retry_after, 0.0), cap)
            else:
                ceiling = min(base * 2 ** attempt, cap)
                delay = ceiling / 2 + random.uniform(0, ceiling / 2)  # "equal jitter": never above the cap
            time.sleep(delay)


def _retry_after(response):
    """Seconds from a Retry-After header (delta-seconds or HTTP date), else None."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return max(parsedate_to_datetime(value).timestamp() - time.time(), 0.0)
        except (TypeError, ValueError):
            return None


def _request(method, url, **kwargs):
    """One HTTP call; transient failures become RetryableError, other HTTP errors raise httpx.HTTPStatusError."""
    try:
        response = httpx.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"network error: {type(exc).__name__}") from exc
    if response.status_code in RETRYABLE_STATUS:
        wait = _retry_after(response)
        if wait is not None and wait > MAX_RETRY_AFTER:
            raise RuntimeError(f"HTTP {response.status_code}: rate limited for {wait:.0f}s, giving up")
        raise RetryableError(f"HTTP {response.status_code}", retry_after=wait)
    response.raise_for_status()
    return response


def _redact(text):
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "<redacted>")
    return re.sub(r"(exaApiKey=)[^&\s\"']+", r"\1<redacted>", text)


def _error(exc):
    return _redact(f"ERROR: {type(exc).__name__}: {exc}")[:400]


def _clean(text):
    return " ".join(str(text or "").split())


# ---- provenance: every url a tool returned in this process; research.py flags sources whose url is not in it ----
SEEN_URLS = set()


def canonical_url(url):
    """Comparable form of a url: no trailing punctuation or slash, lower-case host without www., no arXiv version."""
    url = url.strip().rstrip(".,;:")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:")
    match = re.match(r"https?://(?:www\.)?([^/]+)(.*)", url, re.I)
    if not match:
        return url
    host, path = match.group(1).lower(), match.group(2).rstrip("/")
    if host.endswith("arxiv.org"):
        path = re.sub(r"v\d+(\.pdf)?$", "", path)
    return host + path


def _remember(text):
    """Record the urls of a tool answer, then return the answer unchanged."""
    SEEN_URLS.update(canonical_url(u) for u in re.findall(r"https?://[^\s\"'<>\]]+", text))
    return text


def _clamp(value, low, high):
    return max(low, min(int(value), high))


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()  # researchers run in parallel: serialize arXiv calls to keep the spacing
_arxiv_last_call = 0.0


def _arxiv_get(params):
    global _arxiv_last_call
    with _arxiv_lock:
        wait = _arxiv_last_call + ARXIV_SPACING - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return _request("GET", ARXIV_URL, params=params)
        finally:
            _arxiv_last_call = time.monotonic()


def _arxiv_record(entry):
    raw_id = entry.findtext("a:id", "", ATOM)
    if "/abs/" not in raw_id:  # arXiv reports query errors as an entry pointing at /api/errors
        raise RuntimeError(f"arXiv error: {_clean(entry.findtext('a:summary', '', ATOM))[:200]}")
    paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
    return {"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
            "published": entry.findtext("a:published", "", ATOM)[:10],
            "title": _clean(entry.findtext("a:title", "", ATOM)),
            "summary": _clean(entry.findtext("a:summary", "", ATOM))[:SUMMARY_CHARS]}


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Use a few plain keywords (e.g. "world model video"); all of
    them must match. Returns a JSON list of {id, url, published, title, summary}, "NO RESULTS" or "ERROR: ..."."""
    try:
        terms = [t for t in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query) if t.lower() not in ARXIV_OPERATORS][:8]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
                  "sortOrder": "descending", "start": 0, "max_results": _clamp(max_results, 1, 30)}
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        entries = xml.etree.ElementTree.fromstring(response.content).findall("a:entry", ATOM)
        records = [_arxiv_record(entry) for entry in entries]
        return _remember(json.dumps(records, ensure_ascii=False)) if records else "NO RESULTS"
    except Exception as exc:  # a tool never raises
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_record(item, prefer_ai_summary=False):
    """Map one Hugging Face papers item to a compact record; None when it has no paper id."""
    paper = item.get("paper") or {}
    if not paper.get("id"):
        return None
    summary = (prefer_ai_summary and paper.get("ai_summary")) or paper.get("summary") or item.get("summary")
    return {"id": paper["id"], "url": f"https://huggingface.co/papers/{paper['id']}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(summary)[:SUMMARY_CHARS],
            "upvotes": paper.get("upvotes") or 0, "github": paper.get("githubRepo"),
            "stars": paper.get("githubStars")}


def _hf_get(url, params):
    return with_retry(lambda: _request("GET", url, params=params), attempts=5, base=1.0, cap=30.0).json()


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": _clamp(limit, 1, 100)}
        if date.strip():
            params["date"] = date.strip()
        records = [r for r in map(_hf_record, _hf_get(HF_DAILY_URL, params)) if r]
        words = keyword.lower().split()
        if words:  # every word of the keyword must appear in the title or summary
            records = [r for r in records if all(w in f"{r['title']} {r['summary']}".lower() for w in words)]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return _remember(json.dumps(records, ensure_ascii=False)) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not query.strip():
            return "NO RESULTS"
        items = _hf_get(HF_SEARCH_URL, {"q": query.strip(), "limit": _clamp(limit, 1, 50)})
        records = [r for r in (_hf_record(item, prefer_ai_summary=True) for item in items) if r]
        return _remember(json.dumps(records, ensure_ascii=False)) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _is_rate_limit_flag(meta):
    """True when Exa's result._meta carries a truthy key that mentions a rate limit (any nesting level)."""
    if not isinstance(meta, dict):
        return False
    for key, value in meta.items():
        name = key.lower().replace("_", "").replace("-", "")
        if "ratelimit" in name and value:
            return True
        if _is_rate_limit_flag(value):
            return True
    return False


def _exa_message(response):
    """The JSON-RPC message of an Exa answer (server-sent events or plain JSON)."""
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return response.json()
    events = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
    if not events:
        raise RuntimeError("Exa sent no data event")
    return json.loads(events[-1])


def _exa_call(name, arguments):
    """Call one Exa MCP tool (JSON-RPC tools/call over HTTP) and return its text."""
    headers = {"Accept": "application/json, text/event-stream"}
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:  # a header, not the ?exaApiKey= query parameter: the key then never shows up in httpx error messages
        headers["Authorization"] = f"Bearer {key}"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    def call():
        message = _exa_message(_request("POST", EXA_URL, json=body, headers=headers, timeout=60))
        if "error" in message:
            text = _clean((message["error"] or {}).get("message"))
            if "rate limit" in text.lower():
                raise RetryableError(f"Exa rate limited: {text[:200]}")
            raise RuntimeError(f"Exa error: {text[:300]}")
        result = message.get("result") or {}
        # the free tier signals a rate limit with HTTP 200 and a flag in result._meta, not with HTTP 429
        if _is_rate_limit_flag(result.get("_meta")):
            raise RetryableError("Exa rate limited (flag in result._meta)")
        text = "\n\n".join(c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text").strip()
        if result.get("isError"):
            if "rate limit" in text.lower():
                raise RetryableError(f"Exa rate limited: {text[:200]}")
            raise RuntimeError(f"Exa tool error: {text[:300]}")
        return text

    return with_retry(call, attempts=6, base=2.0, cap=60.0)


def _truncate(text):
    return text if len(text) <= PAGE_CHARS else text[:PAGE_CHARS] + "\n...[truncated]"


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa): blogs, surveys, project pages, docs. `query` = a few keywords; `objective` = one sentence
    describing the ideal page. Returns clean text of the top results, each with Title, URL and Published date."""
    try:
        if not query.strip():
            return "NO RESULTS"
        arguments = {"query": query.strip(), "numResults": _clamp(num_results, 1, 10),
                     "objective": objective.strip() or f"Find authoritative, informative pages about: {query.strip()}"}
        text = _exa_call("web_search_exa", arguments)
        return _remember(_redact(_truncate(text))) if text else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        if not url.strip().startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [url.strip()]})
        if not text:
            return "NO RESULTS"
        _remember(url.strip())  # a page that was really fetched is a real url
        return _remember(_redact(_truncate(text)))
    except Exception as exc:
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")  # titles and pages contain non-ASCII text (e.g. on a Windows console)
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")  # the tools never raise
