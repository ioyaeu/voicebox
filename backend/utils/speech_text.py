"""Speech-friendly text preparation for ``voicebox.speak`` / ``POST /speak``.

Agents hand us markdown — headings, bullet lists, fenced code, tables,
links. Read aloud verbatim, that is noise. ``prepare_speech_text`` strips
the markup down to the prose and, optionally, caps the length on a natural
boundary so a long agent answer becomes a short spoken summary.

Both knobs are off by default; per-client bindings can turn them on for
agents (Claude Code's Stop hook, say) that always pass raw markdown.
"""

from __future__ import annotations

import re

from .chunked_tts import split_text_into_chunks

# Below this the sentence-boundary cut degenerates into word salad.
MIN_MAX_CHARS = 50
MAX_MAX_CHARS = 10000

_FENCED_CODE_RE = re.compile(r"```.*?```", re.S)
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_RULE_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.M)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
_QUOTE_RE = re.compile(r"^\s*>\s?", re.M)
_STRIKE_RE = re.compile(r"~~([^~]+)~~")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"\*([^*\n]+)\*")
# Underscore emphasis only between word boundaries — never inside snake_case.
_UNDERSCORE_RE = re.compile(r"(?<![A-Za-z0-9_])_([^_\n]+)_(?![A-Za-z0-9_])")
_BLANK_LINE_RE = re.compile(r"^[ \t]+$", re.M)


def _strip_markdown_tables(text: str) -> str:
    """Drop table blocks, including tables whose rows do not start with ``|``."""
    lines = text.splitlines()
    dropped: set[int] = set()

    def is_table_row(line: str) -> bool:
        return bool(line.strip()) and "|" in line

    for index, line in enumerate(lines):
        if not _TABLE_SEPARATOR_RE.match(line):
            continue

        start = index - 1
        while start >= 0 and is_table_row(lines[start]):
            start -= 1
        end = index + 1
        while end < len(lines) and is_table_row(lines[end]):
            end += 1
        dropped.update(range(start + 1, end))

    return "\n".join(line for index, line in enumerate(lines) if index not in dropped)


def strip_markdown(text: str) -> str:
    """Reduce markdown to the prose a listener would want to hear.

    Fenced code and tables are dropped outright (there is no good way to
    speak them); links keep their label; emphasis, headings, bullets and
    quote markers are unwrapped. Plain text passes through unchanged.
    """
    text = _FENCED_CODE_RE.sub(" ", text)
    text = _strip_markdown_tables(text)
    text = _RULE_RE.sub("", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _UNDERSCORE_RE.sub(r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = _BLANK_LINE_RE.sub("", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def truncate_for_speech(text: str, max_chars: int) -> str:
    """Keep the longest leading run of *text* that fits in *max_chars*.

    Delegates the boundary search to the TTS chunker so the cut lands where
    a chunk boundary would — sentence end, then clause, then whitespace —
    and never inside ``127.0.0.1`` or a ``[tag]``.
    """
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}")
    text = text.strip()
    if len(text) <= max_chars:
        return text
    chunks = split_text_into_chunks(text, max_chars)
    return chunks[0] if chunks else ""


def prepare_speech_text(
    text: str,
    *,
    plain_text: bool = False,
    max_chars: int | None = None,
) -> str:
    """Apply the speech-friendly transforms a caller (or its binding) asked for.

    Returns the text to hand to TTS — possibly empty when stripping left
    nothing to say (a code-only answer, for instance); callers decide how
    to report that.
    """
    if plain_text:
        text = strip_markdown(text)
    if max_chars is not None and text:
        text = truncate_for_speech(text, max_chars)
    return text.strip()
