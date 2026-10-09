"""Offline tests for research.py (fake sandbox, no LLM):   python -m unittest test_research"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import research
from agents import REPORT_PATH, SOURCES_PATH

REPORT = (b"# T\n\n## TL;DR\n- A [1]. B [2]. C [3].\n\n## References\n"
          b"[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-01)\n"
          b"[2] B. hf-search. https://huggingface.co/papers/2502.00002 (2025-02-01)\n"
          b"[3] C. web. https://example.org/c (n.d.)\n")
SOURCES = json.dumps([
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01",
     "source": "arxiv"},
    {"n": 2, "id": "2502.00002", "url": "https://huggingface.co/papers/2502.00002", "title": "B",
     "date": "2025-02-01", "source": "hf-search"},
    {"n": 3, "id": "c", "url": "https://example.org/c", "title": "C", "date": "n.d.", "source": "web"},
]).encode()


def ai(*names, tokens=(10, 5)):
    calls = [{"name": n, "args": {}, "id": f"{n}-{i}", "type": "tool_call"} for i, n in enumerate(names)]
    return AIMessage(content="", tool_calls=calls,
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)})


MESSAGES = [HumanMessage("topic"), ai("write_todos"), ai("task", "task", "task"),
            ToolMessage("done", tool_call_id="task-0"), ai("execute", tokens=(100, 50))]


class SlugifyTest(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(research.slugify("Survey about World Model"), "survey-about-world-model")
        self.assertEqual(research.slugify("../../x"), "x")
        self.assertEqual(research.slugify("  a///b__c  "), "a-b-c")
        self.assertEqual(research.slugify(""), "topic")
        self.assertEqual(research.slugify("../.."), "topic")
        long = research.slugify("word " * 40)
        self.assertLessEqual(len(long), 60)
        self.assertFalse(long.endswith("-"))


class SummarizeTest(unittest.TestCase):
    def test_counts(self):
        summary = research.summarize(MESSAGES, 12.345, "m")
        self.assertEqual(summary, {"model": "m", "elapsed_s": 12.3, "subagent_calls": 3,
                                   "tool_calls": {"execute": 1, "task": 3, "write_todos": 1},
                                   "tokens": {"input": 120, "output": 60}})


class SaveOutputsTest(unittest.TestCase):
    def save(self, files):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch("research.download", return_value=files):
            try:
                path = research.save_outputs(None, "survey about X", MESSAGES, 1.0, "m", reports_dir=tmp)
            except RuntimeError:
                path = None
        return path, tmp

    def test_writes_three_files_byte_for_byte(self):
        path, tmp = self.save({REPORT_PATH: REPORT, SOURCES_PATH: SOURCES})
        self.assertEqual(path, tmp / "survey-about-x.md")
        self.assertEqual(path.read_bytes(), REPORT)
        self.assertEqual((tmp / "survey-about-x.sources.json").read_bytes(), SOURCES)
        meta = json.loads((tmp / "survey-about-x.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["topic"], "survey about X")
        self.assertEqual(meta["subagent_calls"], 3)
        self.assertEqual(meta["n_sources"], 3)
        self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])

    def test_failed_runs_write_nothing(self):
        for files in ({REPORT_PATH: None, SOURCES_PATH: SOURCES}, {REPORT_PATH: b"  \n", SOURCES_PATH: SOURCES},
                      {REPORT_PATH: REPORT, SOURCES_PATH: None}, {REPORT_PATH: REPORT, SOURCES_PATH: b"{not json"},
                      {REPORT_PATH: REPORT, SOURCES_PATH: b"[]"}):
            path, tmp = self.save(files)
            self.assertIsNone(path)
            self.assertEqual(list(tmp.iterdir()), [], files)

    def test_warns_about_grading_risks(self):
        sources = json.loads(SOURCES)
        sources[1]["source"] = "arxiv"  # a Hugging Face url labelled arxiv
        problems = research._warnings({"subagent_calls": 1, "source_families": ["arxiv", "web"]}, sources,
                                      REPORT.decode())
        self.assertEqual(len([p for p in problems if "subagent_calls" in p or "families" in p]), 2)
        self.assertTrue(any("does not match source" in p for p in problems))


    def test_warns_about_urls_no_tool_returned(self):
        sources = json.loads(SOURCES)
        seen = {research.canonical_url(s["url"]) for s in sources[:2]}
        problems = research._warnings({"subagent_calls": 3, "source_families": ["arxiv", "hf-search", "web"]},
                                      sources, REPORT.decode(), seen)
        self.assertEqual(problems, [f"source [3] url {sources[2]['url']} was never returned by a tool "
                                    "(invented or edited url?)"])


class MainTest(unittest.TestCase):
    def test_no_topic(self):
        self.assertEqual(research.main("   "), 2)

    def test_errors_exit_1_and_write_nothing(self):
        with mock.patch("research.make_model", side_effect=RuntimeError("no model")):
            self.assertEqual(research.main("x"), 1)


if __name__ == "__main__":
    unittest.main()
