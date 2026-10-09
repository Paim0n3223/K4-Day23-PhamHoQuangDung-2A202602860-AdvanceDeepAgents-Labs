"""Tests for check_citations.check (standard library only):   python -m unittest test_check_citations"""
import unittest

from check_citations import check
from finalize_citations import finalize

A = "https://arxiv.org/abs/2501.00001"
B = "https://huggingface.co/papers/2502.00002"
C = "https://example.org/page"


def src(n, url, source="arxiv"):
    return {"n": n, "id": str(n), "url": url, "title": f"Paper {n}", "date": "2025-01-01", "source": source}


SOURCES = [src(1, A), src(2, B, "hf-search"), src(3, C, "web")]
REFS = f"## References\n[1] Paper 1. arxiv. {A} (2025-01-01)\n[2] Paper 2. hf-search. {B} (2025-01-01)\n" \
       f"[3] Paper 3. web. {C} (2025-01-01)\n"


def report(body, refs=REFS):
    return f"# T\n\n## TL;DR\n{body}\n\n{refs}"


class CheckTest(unittest.TestCase):
    def assertProblem(self, problems, fragment):
        self.assertTrue(any(fragment in p for p in problems), f"{fragment!r} not in {problems}")

    def test_valid_report(self):
        self.assertEqual(check(report("X [1]. Y [2][3]."), SOURCES), [])

    def test_grouped_citations_count(self):
        self.assertEqual(check(report("X [1, 2]. Y [1-3]."), SOURCES), [])

    def test_empty_or_bad_sources(self):
        self.assertEqual(check(report("X [1]."), []), ["no sources in sources.json"])
        self.assertEqual(check(report("X [1]."), {"n": 1}), ["sources.json must be a JSON list"])

    def test_source_fields(self):
        bad = [src("1", A), src(2, "ftp://x"), src(3, A)]
        problems = check(report("X [1][2][3]."), bad)
        self.assertProblem(problems, "is not an integer")
        self.assertProblem(problems, "does not start with http")
        self.assertProblem(problems, "duplicates entry #")

    def test_missing_references_heading(self):
        self.assertProblem(check("# T\n\nX [1][2][3].\n", SOURCES), "no '## References' heading")

    def test_cited_but_missing_and_never_cited(self):
        problems = check(report("X [1][2][9]."), SOURCES)
        self.assertProblem(problems, "[9] cited but missing")
        self.assertProblem(problems, "source [3] never cited")

    def test_reference_list_numbers_are_not_citations(self):
        self.assertProblem(check(report("X [1][2]."), SOURCES), "source [3] never cited")

    def test_code_and_links_are_not_citations(self):
        problems = check(report("X [1][2]. `arr[3]` and [3](https://example.org/page)"), SOURCES)
        self.assertProblem(problems, "source [3] never cited")

    def test_reference_line_missing_duplicated_or_unknown(self):
        refs = f"## References\n[1] P. {A}\n[1] P. {A}\n[2] P. {B}\n[7] P. https://x.org\n"
        problems = check(report("X [1][2][3].", refs), SOURCES)
        self.assertProblem(problems, "source [3] has no line")
        self.assertProblem(problems, "2 lines for [1]")
        self.assertProblem(problems, "line [7] is not a source")

    def test_bundled_reference_line(self):
        refs = REFS.replace(f"{C} (2025-01-01)", f"{C}; Other. https://other.org/x")
        self.assertProblem(check(report("X [1][2][3].", refs), SOURCES), "holds 2 URLs")

    def test_reference_url_must_match_sources(self):
        refs = REFS.replace(A, "https://arxiv.org/abs/2501.00001v2")
        self.assertProblem(check(report("X [1][2][3].", refs), SOURCES), "!= sources.json url")

    def test_url_with_parentheses(self):
        wiki = "https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)"
        sources = [src(1, wiki, "web")]
        self.assertEqual(check(report("X [1].", f"## References\n[1] Wiki. web. {wiki} (n.d.)\n"), sources), [])

    def test_family_must_match_url(self):
        sources = [src(1, "https://arxiv.org/pdf/2411.14499v3.pdf"), src(2, "https://arxiv.org/abs/2411.14499v3"),
                   src(3, "https://example.org/x", "hf-search"), src(4, "https://example.org/y", "blog")]
        problems = check(report("X [1][2][3][4]."), sources)
        self.assertEqual(sum("does not match source" in p for p in problems), 3)
        self.assertProblem(problems, "'blog' is not one of")

    def test_finalized_report_passes(self):
        messy = "# T\n\n## TL;DR\nX [3, 1]. Y [2-3]. `code [9]`\n\n## References\n[1] junk; more junk\n"
        sources = [src(1, A), src(2, B, "hf-search"), src(3, C, "web"), src(4, A), src(5, "https://unused.org")]
        new_report, new_sources, problems = finalize(messy, sources)
        self.assertEqual(problems, [])
        self.assertEqual(check(new_report, new_sources), [])


if __name__ == "__main__":
    unittest.main()
