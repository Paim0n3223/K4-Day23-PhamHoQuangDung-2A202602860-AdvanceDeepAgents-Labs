"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

# same citation grammar as finalize_citations.py, so the two scripts never disagree
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]; not a Markdown link [3](url)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]")
_URL = re.compile(r"https?://\S+")
# the url must match the family (RUBRIC 2.2): the grader does not count a mislabelled source
_FAMILY_URL = {"arxiv": re.compile(r"https://arxiv\.org/abs/\S+(?<!v\d)(?<!v\d\d)"),
               "hf-daily": re.compile(r"https://huggingface\.co/papers/\S+"),
               "hf-search": re.compile(r"https://huggingface\.co/papers/\S+"),
               "web": re.compile(r"https?://\S+")}


def _group_numbers(group):
    """'1, 3-5' -> [1, 3, 4, 5]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _urls(line):
    """http(s) URLs of a line; trailing punctuation and an unbalanced ")" are not part of the URL."""
    found = []
    for url in _URL.findall(line):
        url = url.rstrip(".,;:")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1].rstrip(".,;:")
        found.append(url)
    return found


def _cited_numbers(body):
    """Numbers cited in the body, ignoring code spans/blocks and Markdown links."""
    cited = set()
    for segment in _CODE.split(body)[::2]:  # odd indexes are code
        for match in _GROUP.finditer(segment):
            cited.update(_group_numbers(match.group(1)))
    return cited


def _check_sources(sources):
    problems, by_n, seen_urls = [], {}, {}
    for i, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources.json entry #{i} is not an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"sources.json entry #{i}: n={n!r} is not an integer")
        elif n in by_n:
            problems.append(f"source number {n} appears twice in sources.json")
        else:
            by_n[n] = entry
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"sources.json entry #{i}: url {url!r} does not start with http:// or https://")
        elif url in seen_urls:
            problems.append(f"sources.json entry #{i}: url {url} duplicates entry #{seen_urls[url]}")
        else:
            seen_urls[url] = i
            family = entry.get("source")
            if family not in _FAMILY_URL:
                problems.append(f"sources.json entry #{i}: source {family!r} is not one of {sorted(_FAMILY_URL)}")
            elif not _FAMILY_URL[family].fullmatch(url):
                problems.append(f"sources.json entry #{i}: url {url} does not match source {family!r} (arxiv needs "
                                "https://arxiv.org/abs/<id> without version, hf-* https://huggingface.co/papers/<id>; "
                                "a page found by web_search/web_fetch is 'web')")
    return problems, by_n


def _check_references(section, by_n):
    problems, lines_of = [], {}
    for line in section.splitlines():
        match = _REF_LINE.match(line)
        if match:
            lines_of.setdefault(int(match.group(1)), []).append(line)
    for n in sorted(set(by_n) - set(lines_of)):
        problems.append(f"source [{n}] has no line in ## References")
    for n, lines in sorted(lines_of.items()):
        if n not in by_n:
            problems.append(f"## References line [{n}] is not a source in sources.json")
            continue
        if len(lines) > 1:
            problems.append(f"## References has {len(lines)} lines for [{n}] (need exactly one)")
        urls = _urls(lines[0])
        if len(urls) != 1:
            problems.append(f"## References line [{n}] holds {len(urls)} URLs (need exactly one): {lines[0].strip()}")
        elif urls[0] != by_n[n].get("url"):
            problems.append(f"## References line [{n}] URL {urls[0]} != sources.json url {by_n[n].get('url')}")
    return problems


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK); the rules are listed in GUIDE.md part 4."""
    if not isinstance(sources, list):
        return ["sources.json must be a JSON list"]
    if not sources:
        return ["no sources in sources.json"]
    problems, by_n = _check_sources(sources)
    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        problems.append("the report has no '## References' heading")
        body, section = report_text, ""
    else:  # the last heading, like finalize_citations.py
        body, section = report_text[:headings[-1].start()], report_text[headings[-1].end():]
    cited = _cited_numbers(body)
    for n in sorted(cited - set(by_n)):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(set(by_n) - cited):
        problems.append(f"source [{n}] never cited in the report body")
    if headings:
        problems += _check_references(section, by_n)
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
