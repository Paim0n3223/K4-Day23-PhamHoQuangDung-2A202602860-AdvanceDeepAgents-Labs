"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (FINALIZER_PATH, NOTES_DIR, RECURSION_LIMIT, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR,
                    build_lead_agent)
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import SEEN_URLS, canonical_url

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            f"Follow your workflow to produce the cited survey report at {REPORT_PATH}, with its sources in "
            f"{SOURCES_PATH}. You are done only when `python3 {VALIDATOR_PATH}` prints OK.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls, tokens = Counter(), {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(sorted(calls.items())), "tokens": tokens}


def _load_sources(raw):
    if raw is None:
        raise RuntimeError(f"the agent did not write {SOURCES_PATH}")
    try:
        sources = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(s, dict) for s in sources):
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list of objects")
    return sources


def _warnings(meta, sources, report_text, seen_urls=None):
    """Grading risks worth a rerun (RUBRIC 2.1, 2.2, 4.1, 4.2). Reported only: the downloaded files are never altered.
    `seen_urls`: canonical urls the tools returned during the run; a source url outside it was typed by the LLM."""
    found = []
    if meta["subagent_calls"] < 3:
        found.append(f"subagent_calls = {meta['subagent_calls']} (need >= 3)")
    if len(meta["source_families"]) < 3:
        found.append(f"source families {meta['source_families']} (need >= 3 of arxiv, hf-daily, hf-search, web)")
    found += [f"citations: {problem}" for problem in check(report_text, sources)]  # includes family/url checks
    if seen_urls is not None:
        found += [f"source [{s.get('n')}] url {s.get('url')} was never returned by a tool (invented or edited url?)"
                  for s in sources if canonical_url(str(s.get("url", ""))) not in seen_urls]
    return found


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, seen_urls=None):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    A missing/empty report or a missing/invalid sources.json raises RuntimeError and writes NOTHING."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = files.get(REPORT_PATH)
    if not report or not report.strip():
        raise RuntimeError(f"the agent produced no report at {REPORT_PATH}")
    sources = _load_sources(files.get(SOURCES_PATH))
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({s["source"] for s in sources if s.get("source")})}
    for warning in _warnings(meta, sources, report.decode("utf-8", errors="replace"), seen_urls):
        print(f"WARNING: {warning}", file=sys.stderr)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    # bytes exactly as downloaded: the submitted report must be the sandbox's version (no newline translation)
    (reports_dir / f"{slug}.sources.json").write_bytes(files[SOURCES_PATH])
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report)
    return report_path


def _describe(call):
    """A one-line view of a lead tool call for the progress log."""
    args = call.get("args") or {}
    detail = (f"{args.get('subagent_type')}: {args.get('description', '')}" if call["name"] == "task"
              else args.get("command") or args.get("file_path") or args.get("path") or "")
    return f"{call['name']} {' '.join(str(detail).split())[:110]}".rstrip()


def run_agent(agent, topic):
    """Run the lead agent to the end, logging its tool calls; return the final state."""
    state, logged = None, 0
    inputs = {"messages": [{"role": "user", "content": build_prompt(topic)}]}
    for state in agent.stream(inputs, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="values"):
        messages = state["messages"]
        for message in messages[min(logged, len(messages)):]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"[lead] {_describe(call)}", file=sys.stderr, flush=True)
        logged = len(messages)
    if state is None:
        raise RuntimeError("the agent produced no output")
    return state


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL")
        start = time.monotonic()
        with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
            made = backend.execute(f"mkdir -p {NOTES_DIR} {WORKDIR}/report")
            if made.exit_code != 0:
                raise RuntimeError(f"cannot create the workspace: {made.output}")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            if backend.execute(f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}").exit_code != 0:
                raise RuntimeError("uploading the validator/finalizer to the sandbox failed")
            print(f"[research] sandbox {backend.id} ready, running the agent on {topic!r} ...", file=sys.stderr)
            SEEN_URLS.clear()
            state = run_agent(build_lead_agent(backend, model), topic)
            report_path = save_outputs(backend, topic, state["messages"], time.monotonic() - start, model_name,
                                       seen_urls=SEEN_URLS)
    except Exception as exc:  # recursion limit, provider/API errors, missing report...: fail loudly, write nothing
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Report saved to {report_path} (with .sources.json and .meta.json)")
    return 0


if __name__ == "__main__":
    sys.stderr.reconfigure(errors="backslashreplace")  # agent text may not fit a legacy console encoding
    sys.exit(main(" ".join(sys.argv[1:])))
