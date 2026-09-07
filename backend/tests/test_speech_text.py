"""Tests for ``utils.speech_text`` — the speech-friendly transforms behind
``voicebox.speak``'s ``plain_text`` / ``max_chars`` (and the matching
per-client binding defaults).

Agents like Claude Code's Stop hook hand over their raw markdown answer;
these pin what a listener ends up hearing.
"""

import pytest

from backend.utils.speech_text import (
    MIN_MAX_CHARS,
    prepare_speech_text,
    strip_markdown,
    truncate_for_speech,
)

# ── strip_markdown ─────────────────────────────────────────────────────────


def test_plain_text_passes_through():
    assert strip_markdown("Deploy complete.") == "Deploy complete."


def test_fenced_code_and_tables_are_dropped():
    text = "Verdict\n\n```bash\nrm -rf /never-read-aloud\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nDone."
    out = strip_markdown(text)
    assert "rm -rf" not in out
    assert "|" not in out
    assert out.splitlines() == ["Verdict", "Done."]


def test_tables_without_leading_pipes_are_dropped():
    text = "Before\n\nName | Value\n---- | -----\nsecret | 42\n\nAfter"
    assert strip_markdown(text) == "Before\nAfter"


def test_links_keep_their_label_and_headings_bullets_unwrap():
    text = "## Result\n\n- see [.mcp.json](.mcp.json)\n1. **bold** and *it*\n> quoted"
    assert strip_markdown(text) == "Result\nsee .mcp.json\nbold and it\nquoted"


def test_tilde_paths_and_snake_case_survive():
    # A blanket [`*_~] strip would turn ~/.claude into /.claude and
    # voicebox_speak into voiceboxspeak — both were real regressions.
    text = "Edit `~/.claude/settings.json`, then call voicebox_speak."
    assert strip_markdown(text) == "Edit ~/.claude/settings.json, then call voicebox_speak."


def test_code_only_answer_strips_to_nothing():
    assert strip_markdown("```py\nprint(1)\n```") == ""


# ── truncate_for_speech ────────────────────────────────────────────────────


def test_short_text_is_untouched():
    assert truncate_for_speech("Short.", 50) == "Short."


def test_cut_lands_on_a_sentence_end_not_inside_an_ip():
    text = "The server listens on 127.0.0.1:17493 and is healthy. " * 6
    out = truncate_for_speech(text, 120)
    assert len(out) <= 120
    assert out.endswith("healthy.")
    assert "127.0.0" not in out.split()[-1]  # no dangling "127.0.0."


def test_cut_falls_back_to_whitespace_without_punctuation():
    out = truncate_for_speech("word " * 100, 60)
    assert len(out) <= 60
    assert not out.endswith("wor")


def test_rejects_absurdly_small_cap():
    with pytest.raises(ValueError, match="at least"):
        truncate_for_speech("anything", MIN_MAX_CHARS - 1)


# ── prepare_speech_text ────────────────────────────────────────────────────


def test_defaults_are_a_no_op():
    text = "# Keep **everything** as-is\n\n```x```"
    assert prepare_speech_text(text) == text


def test_strip_runs_before_cap_so_markup_does_not_eat_the_budget():
    text = "```\n" + "x" * 500 + "\n```\nOnly this sentence should be heard."
    out = prepare_speech_text(text, plain_text=True, max_chars=80)
    assert out == "Only this sentence should be heard."


def test_empty_result_is_returned_not_raised():
    assert prepare_speech_text("```\ncode\n```", plain_text=True, max_chars=100) == ""
