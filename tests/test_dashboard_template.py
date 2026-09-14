"""The dashboard's JavaScript survives the Python string it lives in.

`DASHBOARD_HTML` is an ordinary triple-quoted string, not a raw one, so Python
resolves escape sequences in it before the browser ever sees them. A `\\n`
written for JavaScript has to be spelled `\\\\n` in the source, and spelling it
with one backslash turns it into a real newline inside a JavaScript string
literal -- which cannot span lines. One such error kills the whole `<script>`
block, so the entire dashboard goes blank rather than losing one feature.

That has happened, from exactly one missing backslash. These tests read what
Python actually emits rather than what the source looks like.
"""

from __future__ import annotations

import pytest

from rockfm.dashboard import DASHBOARD_HTML

QUOTES = ("'", '"')


def script() -> str:
    return DASHBOARD_HTML[
        DASHBOARD_HTML.index("<script>") + len("<script>") : DASHBOARD_HTML.rindex(
            "</script>"
        )
    ]


def unterminated_strings(source: str) -> list[tuple[int, str]]:
    """Quoted literals containing a raw newline, as (line number, excerpt).

    Template literals are exempt: backticks may legally span lines, which is
    why they are the right way to write a multi-line message here.
    """
    found: list[tuple[int, str]] = []
    quote: str | None = None
    started_at = 0
    line = 1
    index = 0
    while index < len(source):
        char = source[index]
        if char == "\n":
            if quote in QUOTES:
                found.append((started_at, source[index - 45 : index].strip()))
                quote = None
            line += 1
        elif quote is not None:
            if char == "\\":
                index += 1              # whatever follows is escaped
            elif char == quote:
                quote = None
        elif char in ("'", '"', "`"):
            quote, started_at = char, line
        elif char == "/" and index + 1 < len(source):
            following = source[index + 1]
            if following == "/":
                index = source.find("\n", index)
                if index == -1:
                    break
                continue
            if following == "*":
                end = source.find("*/", index)
                line += source.count("\n", index, end)
                index = end + 2
                continue
        index += 1
    return found


def test_no_javascript_string_spans_a_newline() -> None:
    offenders = unterminated_strings(script())
    assert not offenders, "\n".join(
        f"line {line}: ...{excerpt}" for line, excerpt in offenders
    )


def test_the_scanner_would_catch_it() -> None:
    """A guard nobody has seen fail is not a guard."""
    broken = "const ok = confirm('first line\nsecond line');"
    assert unterminated_strings(broken)

    # The legal ways to write the same thing must stay legal.
    assert not unterminated_strings("const ok = confirm('first\\nsecond');")
    assert not unterminated_strings("const ok = confirm(`first\nsecond`);")


@pytest.mark.parametrize("marker", ("<script>", "</script>", "id=\"reanalyze\""))
def test_the_page_still_carries_its_parts(marker: str) -> None:
    assert marker in DASHBOARD_HTML
